import os
import openai

from gptauto import GPTAuto

class GPT3(GPTAuto):
    def __init__(self, model_name='babbage', api_key=None):
        super().__init__(model_name, api_key)
        if api_key == None:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            openai.api_key = api_key

    def generate(self, prompt, generate_num=32, temperature=1.0, top_p=1.0, repetition_penalty=0.0, stop_sequences=None):
            try:
                output = openai.Completion.create(
                    engine = self.model_name,
                    prompt = prompt,
                    max_tokens = generate_num,
                    temperature = temperature,
                    top_p = top_p,
                    frequency_penalty = repetition_penalty,
                    stop = stop_sequences
                )
                text = output.choices[0].text
            except openai.error.AuthenticationError:
                print('Invalid OpenAI API Key')
                text = ''

            return text