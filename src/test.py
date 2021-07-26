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

model = "gpt-neo-125M"

request_body_model_init = {
    "model": model
}

request_body = {
    "model": model,
    "prompt": "The touhou project",
    "generate_num": 32,
    "temperature": 0.1,
    "top_p": 1.0,
    "repetition_penalty": 3.0
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