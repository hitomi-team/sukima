import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from gptauto import GPTAuto

from transformers import (
        LogitsProcessorList,
        NoBadWordsLogitsProcessor,

        TemperatureLogitsWarper,
        TopPLogitsWarper,
        TopKLogitsWarper,

        StoppingCriteriaList,
        MaxTimeCriteria,
        MaxLengthCriteria
)

from warpers import (
        TailFreeSamplingLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        PhraseBiasProcessor
)

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
                        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=None, config=model_cfg, state_dict=Checkpoint(model_name, self.device))
                else:
                        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                if parallelize:
                        self.model.parallelize()

        # gen_args, generation arguments [ max_length, max_time ]
        # sample_args, sampling arguments [ temp:float, top_p:float, top_k:int, tfs:float, rep_p:float, rep_p_range:int, rep_p_slope:float, bad_words:list, bias_words:list, bias:float ]
        # Example format { "prompt": "A dog crossed the street", "gen_args": {"max_length":40}, "sample_args": {"temp": 0.53, "tfs": 0.993, "rep_p": 3.65, "bias_words": ["wow", "ok"], "bias": 2.3} }
        def generate(self, args):
                logits_warpers = []
                logits_processors = []
                stopping_criterion = []

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
                if "max_length" in args["gen_args"]:
                        if not isinstance(args["gen_args"]["max_length"], int) or args["gen_args"]["max_length"] < 0:
                                raise TypeError("max_length must be a positive integer")
                        prompt_length = len(self.tokenizer.encode(args["prompt"]))
                        stopping_criterion.append(MaxLengthCriteria(args["gen_args"]["max_length"] + prompt_length))
                if "max_time" in args["gen_args"]:
                        if not isinstance(args["gen_args"]["max_time"], float) or args["gen_args"]["max_time"] < 0.0:
                                raise TypeError("max_time must be a positive float")
                        stopping_criterion.append(MaxTimeCriteria(args["gen_args"]["max_time"]))
                if len(stopping_criterion) == 0:
                        raise ValueError("Generation arguments must contain at least one stopping criteria such as max_length or max_time.")
                
                # Warpers

                if "temp" in args["sample_args"]:
                        if not isinstance(args["sample_args"]["temp"], float) or (args["sample_args"]["temp"] < 0.0):
                                raise ValueError("Temperature must be a float greater than 0.0")
                        logits_warpers.append(TemperatureLogitsWarper(args["sample_args"]["temp"]))
                if "top_p" in args["sample_args"]:
                        if not isinstance(args["sample_args"]["top_p"], float) or (args["sample_args"]["top_p"] < 0.0 or args["sample_args"]["top_p"] > 1.0):
                                raise ValueError("top_p must be a float between 0 and 1")
                        logits_warpers.append(TopPLogitsWarper(args["sample_args"]["top_p"]))
                if "top_k" in args["sample_args"]:
                        if not isinstance(args["sample_args"]["top_k"], int) or (args["sample_args"]["top_k"] <= 0):
                                raise ValueError("top_k must be a positive integer")
                        logits_warpers.append(TopKLogitsWarper(args["sample_args"]["top_k"]))
                if "tfs" in args["sample_args"]:
                        if not isinstance(args["sample_args"]["tfs"], float) or (args["sample_args"]["tfs"] < 0.0 or args["sample_args"]["tfs"] > 1.0):
                                raise ValueError("tfs must be a float between 0 and 1")
                        logits_warpers.append(TailFreeSamplingLogitsWarper(args["sample_args"]["tfs"]))
                
                # Processors

                if "rep_p" in args["sample_args"]:
                        slope = None
                        range = None
                        if "rep_p_slope" in args["sample_args"]:
                                if not isinstance(args["sample_args"]["rep_p_slope"], float) or args["sample_args"]["rep_p_slope"] < 0.0:
                                        raise ValueError("rep_p_slope must be a positive float.")
                                slope = args["sample_args"]["rep_p_slope"]
                        if "rep_p_range" in args["sample_args"]:
                                if not isinstance(args["sample_args"]["rep_p_range"], int) or args["sample_args"]["rep_p_range"] < 0:
                                        raise ValueError("rep_p_range must be a positive integer.")
                                range = args["sample_args"]["rep_p_range"]
                        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=args["sample_args"]["rep_p"], slope=slope, penalize_last=range))

                if "bad_words" in args["sample_args"]:
                        if not isinstance(args["sample_args"]["bad_words"], list):
                                raise ValueError("bad_words must be a list")
                        bad_words_ids = []
                        for bad_word in args["sample_args"]["bad_words"]:
                                if not isinstance(bad_word, str):
                                        raise ValueError("bad_words must be a list of strings")
                                bad_words_ids.append(self.tokenizer.encode(bad_word))
                        logits_processors.append(NoBadWordsLogitsProcessor(bad_words_ids, None))
                if "bias_words" in args["sample_args"]:
                        if not isinstance(args["sample_args"]["bias_words"], list):
                                raise ValueError("bias_words must be a list")
                        if "bias" not in args["sample_args"] or not isinstance(args["sample_args"]["bias"], float):
                                raise KeyError("bias_words requires bias")
                        bias_words_ids = []
                        for bias_word in args["sample_args"]["bias_words"]:
                                if not isinstance(bias_word, str):
                                        raise ValueError("bias_words must be a list of strings")
                                bias_words_ids.append(self.tokenizer.encode(bias_word))
                        logits_processors.append(PhraseBiasProcessor(bias_words_ids, args["sample_args"]["bias"]))

                logits_warper = LogitsProcessorList(logits_warpers)
                logits_processor = LogitsProcessorList(logits_processors)
                stopping_criteria = StoppingCriteriaList(stopping_criterion)

                # Generate

                input_ids = self.tokenizer.encode(args["prompt"], return_tensors='pt').to(self.device)
                outputs = self.model.sample(input_ids=input_ids, logits_warper=logits_warper, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=self.tokenizer.eos_token_id)

                return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

"""
if __name__ == "__main__":
        model = GPTHF(model_name="chpt", sharded=True)
        print(model.generate({"prompt": "Hello world!", "gen_args": {"max_length": 10}, "sample_args": {"temperature": 1.0}}))
"""