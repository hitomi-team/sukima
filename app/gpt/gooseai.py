import math
from typing import Optional
from app.core.config import settings
from app.gpt.autohf import AutoHF
from app.models.soft_prompt import SoftPrompt as SoftPromptModel

from transformers import AutoTokenizer

import openai

class OpenAI(AutoHF):
    def __init__(self, model_name='convo-6b', decoder=True):
        self.model_name = model_name
        self.decoder = decoder

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2') # might wanna check if this is the same as the one used in the model, 20B uses a different tokenizer
        # tokenizer is just going to be used to convert eos_token_id to stop token string

        openai.api_key = settings.OPENAI_API_KEY
        openai.api_base = settings.OPENAI_API_BASE # give users option to use openai if they are financial masochists

        engines = openai.Engine.list()
        for e in engines.data:
            if e.id == self.model_name:
                self.engine = e
                break
        
        if self.engine is None:
            raise ValueError(f'OpenAI/GooseAI engine {self.model_name} not found')

    def generate(self, args, *, db_softprompt: Optional[SoftPromptModel] = None):
        prompt = args.get('prompt', None)
        if prompt is None:
            prompt = '<|endoftext|>'

        # Sample arguments
        
        sample_args = args.get('sample_args', None)
        if sample_args is None:
            raise ValueError('sample_args is required')
        
        temperature = sample_args.get('temperature', 1.0)
        top_p = sample_args.get('top_p', 1.0)
        top_k = sample_args.get('top_k', 128)
        tfs = sample_args.get('tfs', 0.99)
        repetition_penalty = sample_args.get('rep_p', 1.0)

        logit_bias = {}
        bad_words = sample_args.get('bad_words', None)
        if bad_words is not None:
            for bad_word in bad_words:
                bad_word = self.tokenizer.encode(bad_word)
                if len(bad_word) > 1:
                    for bad_word_token_idx in range(len(bad_word)):
                        logit_bias[str(bad_word[bad_word_token_idx])] = -math.sin((math.pi*(bad_word_token_idx/(len(bad_word)-1)))/2) * 100.0
                else:
                    logit_bias[str(bad_word[0])] = -100.0
        
        logit_biases = sample_args.get('logit_biases', None)
        if logit_biases is not None:
            for bias in logit_biases:
                logit_bias[str(bias['id'])] = bias['bias']
        
        # Generation arguments

        gen_args = args.get('gen_args', None)
        if gen_args is None:
            raise ValueError('gen_args is required')

        max_tokens = gen_args.get('max_length', None)
        if max_tokens is None:
            raise ValueError('max_length is required')
        
        stop = gen_args.get('eos_token_id', None)
        if stop is not None:
            stop = self.tokenizer.decode(stop)

        # Generate

        output = {}

        output['output'] = prompt + openai.Completion.create(
            engine = self.engine.id,
            prompt = prompt,
            max_tokens = max_tokens,
            stop = stop,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            tfs = tfs,
            repetition_penalty = repetition_penalty,
            logit_bias = logit_bias,
        ).choices[0].text

        return output
    
    def classify(self, args):
        raise NotImplementedError

    def hidden(self, args):
        raise NotImplementedError
