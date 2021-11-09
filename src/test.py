import signal
import sys
import requests
import json

with open("config.json") as json_file:
    config = json.load(json_file)

def sig_handler(signum, frame):
    print("Received signal: ", signum)
    delete_model()
    cleanup_auth(key)
    sys.exit(0)

model = 'hakurei/gpt-j-random-tinier'

request_body_model_init = {
    "model": model
}

request_body = {
    "model": model,
    "args": {
        "prompt": "west",
        "sample_args": {
            "temp": 0.5,
            "top_p": 1.0,
            "top_k": 140,
            "tfs": 0.9,
            "rep_p": 3.65,
            "rep_p_range": 2048,
            "rep_p_slope": 0.18,
            "bad_words": [
            "nword",
            "wow"
            ],
            "bias_words": [
            "fuck",
            "aahhhh",
            "oops!"
            ],
            "bias": 2.0
        },
        "gen_args": {
            "max_length": 10,
            "max_time": 100.0
        }
    }
}

def auth():
    res = requests.get('http://localhost:8080/v1/create_key', headers={'Authorization': config["auth_admin_token"]})
    print(res.json())
    return res.json()["key"]

def cleanup_auth(key):
    request_body = {
        "key": key
    }
    res = requests.post('http://localhost:8080/v1/delete_key', headers={'Authorization': config["auth_admin_token"]}, json=request_body)
    print(res.json())

def does_model_exist(key):
    r = requests.get("http://localhost:8080/v1/models", headers={'Authorization': key})
    for i in r.json()["models"]:
        if i == model:
            return True
    return False

def delete_model(key):
    res = requests.post('http://localhost:8080/v1/delete', json=request_body_model_init, headers={'Authorization': key})
    print(res.json())

def load_model(key):
    res = requests.post('http://localhost:8080/v1/load', json=request_body_model_init, headers={'Authorization': key})
    print(res.json())

key = auth()

if does_model_exist(key) != True:
    load_model(key)

while True:
    if does_model_exist(key):
        try:
            res = requests.post('http://localhost:8080/v1/generate', json=request_body, headers={'Authorization': key})
            print(res.json()["completion"]["text"], '\n')
        except KeyboardInterrupt:
            if does_model_exist(key):
                delete_model(key)
                break
    else:
        break

cleanup_auth(key)