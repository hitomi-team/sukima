from flask import jsonify, request
from gptauto import GPTAuto
from gpthf import GPTHF
from util import Util
from functools import wraps
from auth import Auth

class API:
    def __init__(self, version='v1', app=None, config=None):
        self.app = app
        self.version = version
        self.config = config
        if config["auth_enable"]:
            self.auth = Auth(config)
        self.models = []
    
    # Auth and Administration
    def verify_admin_token(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            self = args[0]
            token = None
            if self.config["auth_enable"] == True:
                if 'Authorization' in request.headers:
                    token = request.headers['Authorization']
                else:
                    return Util.error(None, "No token header")
                if token != self.config["auth_admin_token"]:
                    return Util.error(None, "Invalid token")
            return f(*args, **kwargs)
        return decorated_function

    def verify_user_token(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            self = args[0]
            token = None
            if self.config["auth_enable"] == True:
                if 'Authorization' in request.headers:
                    token = request.headers['Authorization']
                else:
                    return Util.error(None, "No token header")
                if (self.auth.auth_key(token) == False) and (token != self.config["auth_admin_token"]):
                    return Util.error(None, "Invalid token")
            return f(*args, **kwargs)
        return decorated_function

    # /v1/create_key
    @verify_admin_token
    def create_key(self):
        if self.config["auth_enable"] == False:
            return Util.error(None, "Authentication disabled")
        return jsonify({
            "key": self.auth.create_key()
        })
    
    # /v1/delete_key
    @verify_admin_token
    def delete_key(self):
        request_body = request.json
        if request_body == None:
            return Util.error(None, "No request body")
        if self.config["auth_enable"] == False:
            return Util.error(None, "Authentication disabled")
        
        self.auth.delete_key(request_body["key"])
        return Util.success("Key deleted")

    # API Functions

    # Get models
    # /v1/models
    @verify_user_token
    def get_model_list(self):
        model_dict = {
            "models": {
            }
        }

        for model in self.models:
            model_dict["models"][model.model_name] = {'ready': True}

        return jsonify(model_dict)
    
    # Load model
    # /v1/load
    @verify_user_token
    def load_model(self):
        request_body = request.json
        if request_body == None:
            return Util.error(None, "No request body")
        
        # Check that model exists
        if self.models != None:
            for m in self.models:
                if m.model_name == request_body["model"]:
                    return Util.error(None, "Model already loaded")
        
        if "model" not in request_body:
            return Util.error(None, "No model specified")

        try:
            parallel=False
            if "parallel" in request_body:
                parallel = request_body["parallel"]
            model = GPTHF(model_name=request_body["model"], parallelize=parallel)
            self.models.append(model)
            return Util.success("Loaded model")
        except:
            return Util.error(None, "Unsupported model")
    
    # Delete model
    # /v1/delete
    @verify_user_token
    def delete_model(self):
        request_body = request.json
        if request_body == None:
            return Util.error(None, "No request body")
        
        for m in self.models:
            if m.model_name == request_body["model"]:
                self.models.remove(m)
                return Util.success("Deleted model")
        
        return Util.error(None, "Model not found")

    # Generate from model
    # /v1/generate
    @verify_user_token
    def generate(self):
        request_body = request.json
        if request_body == None:
            return Util.error(None, "No request body")
        
        for m in self.models:
            if m.model_name == request_body["model"]:
                try:
                    if "args" not in request_body:
                        return Util.error(None, "No generation arguments specified")
                    return Util.completion(m.generate(request_body["args"]))
                except Exception as e:
                    return Util.error(None, 'Invalid request body!')

        return Util.error(None, "Model not found")
    
    def run(self):
        self.app.add_url_rule('/{}/create_key'.format(self.version), 'create_key', self.create_key, methods=['GET'])
        self.app.add_url_rule('/{}/delete_key'.format(self.version), 'delete_key', self.delete_key, methods=['POST'])
        self.app.add_url_rule('/{}/models'.format(self.version), view_func=self.get_model_list, methods=['GET'])
        self.app.add_url_rule('/{}/generate'.format(self.version), view_func=self.generate, methods=['POST'])
        self.app.add_url_rule('/{}/load'.format(self.version), view_func=self.load_model, methods=['POST'])
        self.app.add_url_rule('/{}/delete'.format(self.version), view_func=self.delete_model, methods=['POST'])
