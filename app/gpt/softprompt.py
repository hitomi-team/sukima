import torch
import base64
import json
import zlib
import copy
import numpy as np

from app.gpt.quantization import convert_to_int8
from transformers import AutoModelForCausalLM

# globals
current_sp = None

class SoftPrompt():
    def __init__(self, softembedding: torch.tensor = None, n_tokens: int = 20):
        if softembedding is None:
            raise ValueError('softembeddings must not be None')
        if n_tokens is None or n_tokens <= 0:
            raise ValueError('n_tokens must be a positive int')
        
        self.softembedding = softembedding
        self.n_tokens = n_tokens

    def get_input_embeds(self):
        return self.softembedding
    
    def get_special_ids(self, n_vocab = 50257):
        ids = []
        for i in range(self.n_tokens):
            ids.append(n_vocab + i)
        return ids
    
    def get_special_str(self):
        ids = ''
        for i in range(self.n_tokens):
            ids = ids + f'<softtoken_{i}>'
        return ''.join(ids)

    def get_tokenizer(self, tokenizer):
        sp_tokenizer = copy.deepcopy(tokenizer)
        for i in range(self.n_tokens):
            sp_tokenizer.add_tokens(f'<softtoken_{i}>')
        return sp_tokenizer

def resize_model_embeddings(_model, _tokenizer):
    _model.resize_token_embeddings(len(_tokenizer))

class GPTSoftPromptMixin:
    def replace_special_tokens(self, input_ids):
        input_embeds = self.transformer.wte(input_ids.to(self.device))

        if current_sp is None:
            return input_embeds
        
        n_batches = input_ids.shape[0]
        n_tokens = input_ids.shape[-1]
        sp_tokens = current_sp.get_special_ids()

        for b in range(n_batches):
            for t in range(n_tokens):
                input_id = input_ids[b,t].item()
                if input_id in sp_tokens:
                    replacement = current_sp.get_input_embeds().to(self.device).clone().unsqueeze(0)
                    input_embeds[b,t:t+len(sp_tokens[input_id]),:] = replacement[0,:,:]
        
        return input_embeds

    def forward(self, *args, **kwargs):
        if kwargs.get('input_ids') is None:
            kwargs['input_ids'] = args[0]

        if kwargs.get('input_ids') is None:
            return super().forward(*args, **kwargs)
        
        kwargs['input_ids'] = None
        kwargs['input_embeds'] = self.replace_special_tokens(kwargs.get('input_ids'))

        args = ()

        return super().forward(*args, **kwargs)

class AutoModelForSoftPromptLM(GPTSoftPromptMixin, AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if 'quantized' in config.__dict__:
            if config.quantized:
                convert_to_int8(self)
