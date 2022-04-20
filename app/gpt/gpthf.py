from typing import Optional
import torch

from app.core.config import settings
from app.core.logging import logger
from app.gpt.autohf import AutoHF
from app.gpt.softprompt import SoftPrompt, AutoModelForSoftPromptLM, current_sp, resize_model_embeddings
from app.gpt.tensorize import tensorize, untensorize
from app.gpt.utils import Checkpoint, get_dtype, tensorized_path
from app.gpt.warpers import *
from app.models.soft_prompt import SoftPrompt as SoftPromptModel
from transformers import (AutoConfig, AutoTokenizer,
                          LogitsProcessorList, MaxLengthCriteria,
                          MaxTimeCriteria, NoBadWordsLogitsProcessor,
                          StoppingCriteriaList, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper, MinLengthLogitsProcessor)

from pathlib import Path

import numpy as np
import zlib

try:
    import transformers
    from app.gpt.quantization import GPTJBlock, GPTJForCausalLM
except ImportError:
    pass # don't do quantization

class GPTHF(AutoHF):
    def __init__(self, model_name='hakurei/gpt-j-random-tinier', device=None, parallelize=False, sharded=False, quantized=False, tensorized=False):
        super().__init__(model_name=model_name, decoder=True)
        
        model_dtype = get_dtype(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tensorized = False

        if tensorized:
            # check if tensorized model already exists so we can skip expensive model loading below
            _path, exists = tensorized_path(model_name)
            if exists:
                logger.info(f'Loading tensorized model {model_name}')
                self.model = untensorize(str(_path), self.device, quantized=quantized)
                self.tensorized = True

        if sharded:
            model_cfg = AutoConfig.from_pretrained(model_name, return_dict_in_generate=True)
            self.model = AutoModelForSoftPromptLM.from_pretrained(
                pretrained_model_name_or_path=None, config=model_cfg, state_dict=Checkpoint(model_name, self.device), torch_dtype=model_dtype
            ).eval().to(self.device)
        elif (not sharded) and (not quantized) and (not self.tensorized):
            self.model = AutoModelForSoftPromptLM.from_pretrained(model_name, return_dict_in_generate=True, torch_dtype=model_dtype).eval().to(self.device)

        if quantized:
            self.quantized = True
            logger.info(f'Quantizing model {model_name}')
            # we assume this is a gptj model - TODO: fix this
            transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J
            if not self.tensorized:
                self.model = AutoModelForSoftPromptLM.from_pretrained(model_name, low_cpu_mem_usage=True, return_dict_in_generate=True).eval().to(self.device)
            logger.info(f'Quantization complete.')
        else:
            self.quantized = False

        if (tensorized) and (not self.tensorized):
            # check if model file exists in ./storage/{model_name}.model
            _path, exists = tensorized_path(model_name)
            if not exists:
                logger.info(f'Tensorizing model {model_name}')
                # tensorize model
                tensorize(self.model, str(_path))
                del self.model
                raise Exception('Tensorized the model! The original model has been altered, please load the model again to use the tensorized model.')

        if parallelize:
            self.model.parallelize()

    @torch.inference_mode()
    def generate(self, args, *, db_softprompt: Optional[SoftPromptModel] = None):
        logits_warpers = []
        logits_processors = []
        stopping_criterion = []
        eos_token_id = None
        softprompt = None
        output_scores = False
        best_of = None
        prompt_length = None

        # Check if args are valid since it's a dictionary
        if not isinstance(args, dict):
            raise TypeError("Arguments must be a dictionary")

        if db_softprompt:
            tbuf = np.frombuffer(zlib.decompress(db_softprompt.read()), dtype=np.float16)
            tensor = torch.from_numpy(np.array(tbuf).reshape(20, len(tbuf)//20)).to(self.device)
            softprompt = SoftPrompt(softembedding=tensor)
            sp_ids = [[id] for id in softprompt.get_special_ids()]
            logits_processors.append(NoBadWordsLogitsProcessor(sp_ids, None))

        if "prompt" not in args:
            raise KeyError("Arguments must contain a prompt")
        else:
            if softprompt:
                prompt = softprompt.get_special_str() + args["prompt"]
            else:
                prompt = args["prompt"]

        if "gen_args" not in args:
            raise KeyError("Arguments must contain generation arguments")

        if "sample_args" not in args:
            raise KeyError("Arguments must contain sampling arguments")

        # Stopping criteria
        if "max_length" in args["gen_args"] and args["gen_args"]["max_length"]:
            if not isinstance(args["gen_args"]["max_length"], int) or args["gen_args"]["max_length"] < 0:
                raise TypeError("max_length must be a positive integer")

            prompt_length = len(self.tokenizer.encode(args["prompt"]))
            if softprompt:
                prompt_length += 20
            stopping_criterion.append(MaxLengthCriteria(args["gen_args"]["max_length"] + prompt_length))

        if "max_time" in args["gen_args"] and args["gen_args"]["max_time"]:
            if not isinstance(args["gen_args"]["max_time"], float) or args["gen_args"]["max_time"] < 0.0:
                raise TypeError("max_time must be a positive float")

            stopping_criterion.append(MaxTimeCriteria(args["gen_args"]["max_time"]))
        
        if "eos_token_id" in args["gen_args"] and args["gen_args"]["eos_token_id"]:
            if not isinstance(args["gen_args"]["eos_token_id"], int) or args["gen_args"]["eos_token_id"] < 0:
                raise TypeError("eos_token_id must be a positive integer")

            eos_token_id = args["gen_args"]["eos_token_id"]

        if "min_length" in args["gen_args"] and args["gen_args"]["min_length"]:
            if not isinstance(args["gen_args"]["min_length"], int) or args["gen_args"]["min_length"] > args["gen_args"]["max_length"]:
                raise TypeError("min_length must be an integer less than max_length.")

            logits_processors.append(MinLengthLogitsProcessor(args["gen_args"]["min_length"], eos_token_id))

        if "logprobs" in args["gen_args"] and args["gen_args"]["logprobs"]:
            if not isinstance(args["gen_args"]["logprobs"], int) or args["gen_args"]["logprobs"] < 0 or args["gen_args"]["logprobs"] > 20:
                raise TypeError("logprobs must be an integer between 0 and 20.")
            output_scores = True

        if "best_of" in args["gen_args"] and args["gen_args"]["best_of"]:
            if not isinstance(args["gen_args"]["best_of"], int) or args["gen_args"]["best_of"] < 0:
                raise TypeError("best_of must be a positive integer.")
            best_of = args["gen_args"]["best_of"]
            output_scores = True

        if len(stopping_criterion) == 0:
            raise ValueError("Generation arguments must contain at least one stopping criteria such as max_length or max_time.")

        # Warpers
        if "temp" in args["sample_args"] and args["sample_args"]["temp"]:
            if not isinstance(args["sample_args"]["temp"], float) or (args["sample_args"]["temp"] < 0.0):
                raise ValueError("Temperature must be a float greater than 0.0")

            logits_warpers.append(TemperatureLogitsWarper(args["sample_args"]["temp"]))

        if "top_p" in args["sample_args"] and args["sample_args"]["top_p"]:
            if not isinstance(args["sample_args"]["top_p"], float) or (args["sample_args"]["top_p"] < 0.0 or args["sample_args"]["top_p"] > 1.0):
                raise ValueError("top_p must be a float between 0 and 1")

            logits_warpers.append(TopPLogitsWarper(args["sample_args"]["top_p"]))

        if "top_k" in args["sample_args"] and args["sample_args"]["top_k"]:
            if not isinstance(args["sample_args"]["top_k"], int) or (args["sample_args"]["top_k"] <= 0):
                raise ValueError("top_k must be a positive integer")

            logits_warpers.append(TopKLogitsWarper(args["sample_args"]["top_k"]))

        if "top_a" in args["sample_args"] and args["sample_args"]["top_a"]:
            if not isinstance(args["sample_args"]["top_a"], float) or (args["sample_args"]["top_a"] < 0.0 or args["sample_args"]["top_a"] > 1.0):
                raise ValueError("top_a must be a float between 0 and 1")

            logits_warpers.append(TopALogitsWarper(args["sample_args"]["top_a"]))

        if "typical_p" in args["sample_args"] and args["sample_args"]["typical_p"]:
            if not isinstance(args["sample_args"]["typical_p"], float) or (args["sample_args"]["typical_p"] < 0.0 or args["sample_args"]["typical_p"] > 1.0):
                raise ValueError("typical_p must be a float between 0 and 1")

            logits_warpers.append(TypicalLogitsWarper(args["sample_args"]["typical_p"]))

        if "tfs" in args["sample_args"] and args["sample_args"]["tfs"]:
            if not isinstance(args["sample_args"]["tfs"], float) or (args["sample_args"]["tfs"] < 0.0 or args["sample_args"]["tfs"] > 1.0):
                raise ValueError("tfs must be a float between 0 and 1")

            logits_warpers.append(TailFreeSamplingLogitsWarper(args["sample_args"]["tfs"]))

        # Processors
        if "rep_p" in args["sample_args"] and args["sample_args"]["rep_p"]:
            rep_slope = None
            rep_range = None

            if "rep_p_slope" in args["sample_args"] and args["sample_args"]["rep_p_slope"]:
                if not isinstance(args["sample_args"]["rep_p_slope"], float) or args["sample_args"]["rep_p_slope"] < 0.0:
                    raise ValueError("rep_p_slope must be a positive float.")

                rep_slope = args["sample_args"]["rep_p_slope"]

            if "rep_p_range" in args["sample_args"] and args["sample_args"]["rep_p_range"]:
                if not isinstance(args["sample_args"]["rep_p_range"], int) or args["sample_args"]["rep_p_range"] < 0:
                    raise ValueError("rep_p_range must be a positive integer.")

                rep_range = args["sample_args"]["rep_p_range"]

            logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=args["sample_args"]["rep_p"], slope=rep_slope, penalize_last=rep_range))

        if "bad_words" in args["sample_args"] and args["sample_args"]["bad_words"]:
            if not isinstance(args["sample_args"]["bad_words"], list):
                raise ValueError("bad_words must be a non-empty list")

            bad_words_ids = []

            for bad_word in args["sample_args"]["bad_words"]:
                if not isinstance(bad_word, str):
                    raise ValueError("bad_words must be a list of strings")

                bad_words_ids.append(self.tokenizer.encode(bad_word))

            logits_processors.append(NoBadWordsLogitsProcessor(bad_words_ids, None))

        if "logit_biases" in args["sample_args"] and args["sample_args"]["logit_biases"]:
            if not isinstance(args["sample_args"]["logit_biases"], list):
                raise ValueError("logit_biases must be a list")
            
            logit_biases = []

            for logit_bias in args["sample_args"]["logit_biases"]:
                if not isinstance(logit_bias, dict) or "id" not in logit_bias or "bias" not in logit_bias:
                    raise ValueError("logit_biases must be a list of dicts with keys 'id' and 'bias'")

                if not isinstance(logit_bias["id"], int):
                    raise ValueError("logit_biases 'id' must be an integer")

                if not isinstance(logit_bias["bias"], float):
                    raise ValueError("logit_biases 'bias' must be a float")

                logit_biases.append((logit_bias["id"], logit_bias["bias"]))
            
            logits_processors.append(LogitBiasProcessor(logit_biases))

        if "phrase_biases" in args["sample_args"] and args["sample_args"]["phrase_biases"]:
            if not isinstance(args["sample_args"]["phrase_biases"], list):
                raise ValueError("phrase_biases must be a non-empty list")
            
            for bias in args["sample_args"]["phrase_biases"]:
                if not isinstance(bias, dict):
                    raise ValueError("biases must be a list of dictionaries")

                if "sequences" not in bias or not isinstance(bias["sequences"], list):
                    raise ValueError("phrase_biases must be a list of dictionaries with sequences")

                if "bias" not in bias or not isinstance(bias["bias"], float):
                    raise ValueError("biases must be a list of dictionaries with a bias key")

                if "ensure_sequence_finish" not in bias or not isinstance(bias["ensure_sequence_finish"], bool):
                    raise ValueError("biases must be a list of dictionaries with an ensure_sequence_finish key")

                if "generate_once" not in bias or not isinstance(bias["generate_once"], bool):
                    raise ValueError("biases must be a list of dictionaries with a generate_once key")

                logits_processors.append(PhraseBiasProcessor([self.tokenizer.encode(sequence) for sequence in bias["sequences"]], bias["bias"], bias["ensure_sequence_finish"], bias["generate_once"]))

        logits_warper = LogitsProcessorList(logits_warpers)
        logits_processor = LogitsProcessorList(logits_processors)
        stopping_criteria = StoppingCriteriaList(stopping_criterion)

        # Generate
        output = {}
        best_of_idx = 0

        global current_sp
        current_sp = softprompt
        if softprompt:
            sp_tokenizer = softprompt.get_tokenizer(self.tokenizer)
            resize_model_embeddings(self.model, sp_tokenizer)
            input_ids = sp_tokenizer(prompt, return_tensors='pt').to(self.device)
        else:
            resize_model_embeddings(self.model, self.tokenizer)
            input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        outputs = None
        if best_of is None:
            outputs = self.model.sample(
                **input_ids,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores
            )
        else:
            best_of_outputs = []
            best_of_sequences = []
            for i in range(best_of):
                outputs = self.model.sample(
                    **input_ids,
                    logits_warper=logits_warper,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=eos_token_id,
                    output_scores=output_scores
                )
                scores = []
                for token_idx in range(len(outputs.sequences[0]) - prompt_length):
                    scores.append(outputs.scores[token_idx][0][outputs.sequences[0][token_idx + prompt_length]].detach().item())
                best_of_sequences.append(torch.tensor(scores).mean().detach().item())
                best_of_outputs.append(outputs)
            best_of_idx = best_of_sequences.index(max(best_of_sequences))
            outputs = best_of_outputs[best_of_idx]

        output["output"] = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
#        if softprompt:
#            output["output"] = output["output"][len(softprompt.get_special_str()):]
            
        if "logprobs" in args["gen_args"] and args["gen_args"]["logprobs"]:
            if not isinstance(args["gen_args"]["logprobs"], int) or args["gen_args"]["logprobs"] < 0 or args["gen_args"]["logprobs"] > 20:
                pass
            logprobs = []
            for i in range(len(outputs.scores)):
                logprobs_seq = []
                scores_probs = outputs.scores[i].softmax(-1).topk(args["gen_args"]["logprobs"], dim=-1).values.tolist()
                scores_indices = outputs.scores[i].topk(args["gen_args"]["logprobs"], dim=-1).indices.tolist()
                for j in range(args['gen_args']['logprobs']):
                    logprobs_seq.append((scores_indices[0][j], scores_probs[0][j]))
                logprobs.append(logprobs_seq)
            output["logprobs"] = logprobs
        
        return output

    @torch.inference_mode()
    def classify(self, args):
        if not isinstance(args, dict):
            raise ValueError("args must be a dictionary")

        if "prompt" not in args or not isinstance(args["prompt"], str):
            raise ValueError("args must contain a prompt")
        
        if "labels" not in args or not isinstance(args["labels"], list):
            raise ValueError("args must contain a list of labels")
        
        for label in args["labels"]:
            if not isinstance(label, str):
                raise ValueError("labels must be a list of integers")

        prompt_inputs = self.tokenizer(args["prompt"], return_tensors='pt').input_ids.to(self.device)

        output_probs = {}
        for i in args["labels"]:
            label_inputs = self.tokenizer(i, return_tensors='pt').input_ids.to(self.device)
            probs = self.model.forward(input_ids=torch.cat([prompt_inputs, label_inputs], dim=-1)).logits.softmax(-1)[0][-len(label_inputs[0]):]
            token_probs = [probs[t][label_inputs[0][t]] for t in range(0, len(label_inputs[0]))]
            output_probs[i] = torch.mean(torch.stack(token_probs, dim=-1)).item() 

        return output_probs
