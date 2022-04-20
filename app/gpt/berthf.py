import torch

from app.core.config import settings
from app.core.logging import logger

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from app.gpt.autohf import AutoHF
from app.gpt.tensorize import tensorize, untensorize
from app.gpt.utils import Checkpoint, get_dtype, tensorized_path

class BERTHF(AutoHF):
    def __init__(self, model_name='distilroberta-base', device=None, parallelize=False, sharded=False, quantized=False, tensorized=False):
        super().__init__(model_name=model_name, decoder=False)

        model_dtype = get_dtype(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tensorized = False

        if tensorized:
            _path, exists = tensorized_path(model_name)
            if exists:
                logger.info(f'Loading tensorized model {model_name}')
                self.model = untensorize(str(_path), self.device, quantized=quantized)
                self.tensorized = True
        
        if sharded:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=None,
                config=AutoConfig.from_pretrained(model_name),
                state_dict=Checkpoint(model_name, self.device),
                torch_dtype=model_dtype
            ).eval().to(self.device)
        
        if (not sharded) and (not quantized) and (not self.tensorized):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=model_dtype
            ).eval().to(self.device)
        
        if quantized:
            raise NotImplementedError('Quantized models are not supported yet for encoder models such as BERT.')
        
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
            raise NotImplementedError('Parallelization is not supported yet for encoder models such as BERT.')
    
    @torch.inference_mode()
    def classify(self, args):
        if not isinstance(args, dict):
            raise ValueError('args must be a dictionary.')
        
        if 'prompt' not in args or not isinstance(args['prompt'], str):
            raise ValueError('args must contain a prompt as a string.')

        
        if "labels" not in args or not isinstance(args["labels"], list):
            raise ValueError("args must contain a list of labels")
        
        for label in args["labels"]:
            if not isinstance(label, str):
                raise ValueError("labels must be a list of integers")
        
        prompt_inputs = self.tokenizer.encode(args['prompt'], return_tensors='pt')

        outputs = self.model(prompt_inputs).logits
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.detach().cpu().numpy()
        output_probs = {}

        # TODO: automatically fill labels

        for i in range(len(args["labels"])):
            output_probs[args["labels"][i]] = float(outputs[0][i])

        return output_probs
