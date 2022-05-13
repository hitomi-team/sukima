import torch
import requests
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.gpt.autohf import AutoHF
from app.gpt.tensorize import tensorize, untensorize
from app.gpt.utils import Checkpoint, get_dtype, tensorized_path

class CLIP(AutoHF):
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None, parallelize=False, sharded=False, quantized=False, tensorized=False):
        super().__init__(model_name=model_name, decoder=False)
        self.device = device
        self.tensorized = False
        processor_name = None
        if not model_name.startswith('openai'):
            processor_name = 'openai/'+model_name
        else:
            processor_name = model_name
        self.processor = CLIPProcessor.from_pretrained(processor_name)

        if tensorized:
            # check if tensorized model already exists so we can skip expensive model loading below
            _path, exists = tensorized_path(model_name)
            if exists:
                logger.info(f'Loading tensorized model {model_name}')
                self.model = untensorize(str(_path), self.device, quantized=quantized)
                self.tensorized = True
        
        if (not quantized) and (not self.tensorized):
            self.model = CLIPModel.from_pretrained(model_name).eval().to(device)

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
    def _text_feats(self, in_text: str):
        text_tokens = self.processor(text=in_text, return_tensors='pt', padding=True)['input_ids'].to(self.device)
        result = self.model.get_text_features(input_ids=text_tokens).cpu().detach().numpy()
        return (result / np.linalg.norm(result, axis=1, keepdims=True)).squeeze(axis=0)
    
    @torch.inference_mode()
    def _img_feats(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')['pixel_values'].to(self.device)
        result = self.model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
        return (result / np.linalg.norm(result)).squeeze(axis=0)
    
    def _sim(self, text_feats, img_feats):
        return np.dot(img_feats, text_feats)/(np.sqrt(np.linalg.norm(img_feats))*np.sqrt(np.linalg.norm(text_feats)))

    def classify(self, args):
        if 'prompt' not in args or not isinstance(args['prompt'], str):
            raise ValueError('args must contain a prompt url as a string')
        if 'labels' not in args or not isinstance(args['labels'], list):
            raise ValueError('args must contain a list of labels')
        for label in args['labels']:
            if not isinstance(label, list):
                raise ValueError('labels must be a list of lists containing strings')
    
        img_feats = self._img_feats(args['prompt'])

        compiled_similarities = []
        for label_set in range(len(args['labels'])):
            label_similarities = {}
            for label in args['labels'][label_set]:
                text_feats = self._text_feats(label)
                label_similarities[label] = self._sim(text_feats, img_feats).item()
            compiled_similarities.append(label_similarities)    
        output = {'labels': compiled_similarities}

        return output
