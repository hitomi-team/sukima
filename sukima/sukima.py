import requests

class sukima:
    def __init__(self, api_key, api_url='https://localhost:8080/v1/', verify=False):
        self.api_key = api_key
        self.api_url = api_url
        self.verify = verify
    
    def key_create(self):
        r = requests.get(self.api_url + 'create_key', headers={'Authorization': self.api_key}, verify=self.verify)
        if "key" not in r.json():
            raise Exception("Key creation failed: ", r.json()["error"]["message"])
        return r.json()["key"]
    
    def key_delete(self, key):
        r = requests.post(self.api_url + 'delete_key', json={"key": key}, headers={'Authorization': self.api_key}, verify=self.verify)
        if "error" in r.json():
            raise Exception("Key deletion failed: ", r.json()["error"]["message"])

    def model_list(self):
        r = requests.get(self.api_url + 'models', headers={'Authorization': self.api_key}, verify=self.verify)
        if "models" not in r.json():
            raise Exception("Get models failed: ", r.json()["error"]["message"])
        return [i for i in r.json()["models"]]

    def model_load(self, model_name):
        return requests.post(self.api_url + 'load', json={"model": model_name}, headers={'Authorization': self.api_key}, verify=self.verify).json()
    
    def model_delete(self, model_name):
        return requests.post(self.api_url + 'delete', json={"model": model_name}, headers={'Authorization': self.api_key}, verify=self.verify).json()
    
    def generate(self, args):
        return requests.post(self.api_url + 'generate', json=args, headers={'Authorization': self.api_key}, verify=self.verify).json()

"""
if __name__ == "__main__":
    api = sukima('admin')

    args = {
        "model": "distilgpt2",
        "args": {
            "prompt": "Sukima is a ready-to-deploy container that serves a REST API for Language Models. Not only does",
            "sample_args": {
                "temp": 0.5,
                "tfs": 0.9,
                "rep_p": 3.65,
                "bias_words": [
                    "GPT is awesome!"
                ],
                "bias": 1.0
            },
            "gen_args": {
                "max_length": 40
            }
        }
    }

    print(api.generate(args)["completion"]["text"])
"""