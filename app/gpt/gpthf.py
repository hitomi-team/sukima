from typing import Type
import torch
from app.gpt.gptauto import GPTAuto
from app.gpt.warpers import *
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, MaxLengthCriteria,
                          MaxTimeCriteria, NoBadWordsLogitsProcessor,
                          StoppingCriteriaList, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper, MinLengthLogitsProcessor)

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

from pathlib import Path


class Checkpoint(MutableMapping):
    def __init__(self, chkpt_dir, device="cpu"):
        self.device = device
        self.chkpt_dir = Path(chkpt_dir)
        self.checkpoint = torch.load(str(chkpt_dir / Path("m.pt")))

    def __len__(self):
        return len(self.checkpoint)

    def __getitem__(self, key):
        path = self.chkpt_dir / Path(self.checkpoint[key]).name

        if self.device == "cpu":
            return torch.load(str(path), map_location=self.device).long()
        else:
            return torch.load(str(path), map_location=self.device).half()

    def __setitem__(self, key, value):
        return

    def __delitem__(self, key, value):
        return

    def keys(self):
        return self.checkpoint.keys()

    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))

    def __copy__(self):
        return Checkpoint(self.chkpt_dir, device=self.device)

    def copy(self):
        return Checkpoint(self.chkpt_dir, device=self.device)


class GPTHF(GPTAuto):
    def __init__(self, model_name='hakurei/gpt-j-random-tinier', device=None, parallelize=False, sharded=False):
        super().__init__(model_name=model_name)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device

        if sharded:
            model_cfg = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=None, config=model_cfg, state_dict=Checkpoint(model_name, self.device), torch_dtype=torch.float16
            ).eval().to(self.device)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if parallelize:
            self.model.parallelize()

    @torch.inference_mode()
    def generate(self, args):
        logits_warpers = []
        logits_processors = []
        stopping_criterion = []
        eos_token_id = None

        # Check if args are valid since it's a dictionary
        if not isinstance(args, dict):
            raise TypeError("Arguments must be a dictionary")

        if "prompt" not in args:
            raise KeyError("Arguments must contain a prompt")

        if "gen_args" not in args:
            raise KeyError("Arguments must contain generation arguments")

        if "sample_args" not in args:
            raise KeyError("Arguments must contain sampling arguments")

        # Stopping criteria
        if "max_length" in args["gen_args"] and args["gen_args"]["max_length"]:
            if not isinstance(args["gen_args"]["max_length"], int) or args["gen_args"]["max_length"] < 0:
                raise TypeError("max_length must be a positive integer")

            prompt_length = len(self.tokenizer.encode(args["prompt"]))
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

        if "tfs" in args["sample_args"] and args["sample_args"]["tfs"]:
            if not isinstance(args["sample_args"]["tfs"], float) or (args["sample_args"]["tfs"] < 0.0 or args["sample_args"]["tfs"] > 1.0):
                raise ValueError("tfs must be a float between 0 and 1")

            logits_warpers.append(TailFreeSamplingLogitsWarper(args["sample_args"]["tfs"]))

        # Processors
        if "rep_p" in args["sample_args"] and args["sample_args"]["rep_p"]:
            slope = None
            range = None

            if "rep_p_slope" in args["sample_args"] and args["sample_args"]["rep_p_slope"]:
                if not isinstance(args["sample_args"]["rep_p_slope"], float) or args["sample_args"]["rep_p_slope"] < 0.0:
                    raise ValueError("rep_p_slope must be a positive float.")

                slope = args["sample_args"]["rep_p_slope"]

            if "rep_p_range" in args["sample_args"] and args["sample_args"]["rep_p_range"]:
                if not isinstance(args["sample_args"]["rep_p_range"], int) or args["sample_args"]["rep_p_range"] < 0:
                    raise ValueError("rep_p_range must be a positive integer.")

                range = args["sample_args"]["rep_p_range"]

            logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=args["sample_args"]["rep_p"], slope=slope, penalize_last=range))

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
        input_ids = self.tokenizer.encode(args["prompt"], return_tensors='pt').to(self.device)
        outputs = self.model.sample(
            input_ids=input_ids,
            logits_warper=logits_warper,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_token_id
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

"""
if __name__ == "__main__":
        model = GPTHF(model_name="chpt", sharded=True)
        print(model.generate({"prompt": "Hello world!", "gen_args": {"max_length": 10}, "sample_args": {"temperature": 1.0}}))
"""
