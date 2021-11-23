![logo](banner.png)

## Overview
Sukima is a ready-to-deploy container that implements a REST API for Language Models designed with the specific purpose of easy deployment and scalability.

### Curent API Functions
- **models** : Fetch a list of ready-to-use Language Models for inference.
- **load** : Allocate a Language Model.
- **delete** : Free a Language Model from memory.
- **generate** : Use a Language Model to generate tokens.

### Setup
1. Customize the ports and host in the [Dockerfile](Dockerfile) to your liking.
2. Install Docker and run ``docker-compose up`` in the directory with the Dockerfile to deploy.
3. That's it!

### Todo
- HTTPS Support
- Rate Limiting
- Support for other Language Modeling tasks such as Sentiment Analysis and Named Entity Recognition.
- Soft Prompt tuning endpoint

### Example API Usage
```python
import json, requests

with open("config.json") as f:
    config = json.load(f)

# Get an API Key
r = requests.get("http://localhost:8080/v1/create_key", headers={"Authorization": config["auth_admin_token"]})
print(r.json())
key = r.json()["key"]

# List currently available models
r = requests.get("http://localhost:8080/v1/models", headers={"Authorization": key})
for i in r.json()["models"]:
  print(i)

# Load the model! In order to do this, we use a simple request body.
request_body_model_init = {"model": "EleutherAI/gpt-neo-125M"}
res = requests.post('http://localhost:8080/v1/load', json=request_body_model_init, headers={"Authorization": key})
print(res.json())

# Generate from the model!
request_body = {
    "model": "EleutherAI/gpt-neo-125M",
    "args": {
        "prompt": "Sukima is a ready-to-deploy container that serves a REST API for Language Models. Not only does",
        "sample_args": {
            "temp": 0.5,
            "tfs": 0.9,
            "rep_p": 3.65,
            "bias_words": [
                "GPT is awesome!"
            ],
            "bias": 2.0
        },
        "gen_args": {
            "max_length": 40
        }
    }
}

res = requests.post('http://localhost:8080/v1/generate', json=request_body, headers={"Authorization": key})
print(res.json()['completion']['text'])

# And finally, delete the model that we have allocated.
res = requests.post('http://localhost:8080/v1/delete', json=request_body_model_init, headers={"Authorization": key})
print(res.json())

# Then, delete the API key.
request_body_delete_key = { "key": key }
r = requests.post('http://localhost:8080/v1/delete_key', json=request_body_delete_key, headers={"Authorization": config["auth_admin_token"]})
print(r.json())
```

### License
[Simplified BSD License](LICENSE)
