from flask import jsonify, request
from gptauto import GPTAuto
from util import Util

class API:
    def __init__(self, version='v1', app=None):
        self.app = app
        self.version = version
    
    def load_models(self, models=[]):
        self.models = models
    
    # API Functions

    # Get models
    # /v1/models
    def get_model_list(self):
        model_dict = {
            "models": {
            }
        }

        for model in self.models:
            print(model.model_name)
            model_dict["models"][model.model_name] = {'ready': True}

        return jsonify(model_dict)
    
    # Generate from model
    # /v1/generate
    def generate(self):
        request_body = request.json
        if request_body == None:
            return Util.error(None, "No request body")
        
        for m in self.models:
            if m.model_name == request_body["model"]:
                try:
                    return Util.completion(m.generate(prompt=request_body["prompt"],
                        generate_num=request_body["generate_num"],
                        temperature=request_body["temperature"],
                        top_p=request_body["top_p"],
                        repetition_penalty=request_body["repetition_penalty"]))
                except Exception as e:
                    return Util.error(None, 'Invalid request body!')

        return Util.error(None, "Model not found")
    
    def run(self):
        self.app.add_url_rule('/{}/models'.format(self.version), view_func=self.get_model_list, methods=['GET', 'POST'])
        self.app.add_url_rule('/{}/generate'.format(self.version), view_func=self.generate, methods=['POST'])
