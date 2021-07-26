class GPTAuto:
        def __init__(self, model_name='generic', api_key=None):
                self.model_name = model_name
                self.api_key = api_key
        
        def generate(self, prompt, generate_num, temperature, top_p, repetition_penalty, stop_sequences):
                max_tokens = 2048 - generate_num
                return prompt