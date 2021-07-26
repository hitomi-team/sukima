from transformers import pipeline
from gptauto import GPTAuto

class GPTNeo(GPTAuto):
        def __init__(self, model_name='gpt-neo-125M', api_key=None):
                super().__init__(model_name=model_name, api_key=api_key)
                self.generator = pipeline('text-generation', model='EleutherAI/' + self.model_name)
        
        def generate(self, prompt, generate_num=32, temperature=1.0, top_p=1.0, repetition_penalty=0.1):
                text = self.generator(prompt, 
                        do_sample=True,
                        min_length=len(prompt),
                        max_length=len(prompt)+generate_num,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=50256)
                return text[0]['generated_text']
