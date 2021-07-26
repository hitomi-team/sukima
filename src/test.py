import signal
import sys
import requests

def sig_handler(signum, frame):
    print("Received signal: ", signum)
    delete_model()
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

def does_model_exist():
    r = requests.get("http://192.168.0.147:8080/v1/models")
    for i in r.json()["models"]:
        if i == model:
            return True
    return False

def delete_model():
    res = requests.post('http://localhost:8080/v1/delete', json=request_body_model_init)
    print(res.json())

def load_model():
    res = requests.post('http://localhost:8080/v1/load', json=request_body_model_init)
    print(res.json())

if does_model_exist() != True:
    load_model()

while True:
    if does_model_exist():
        try:
            res = requests.post('http://localhost/v1/generate', json=request_body)
            print(res.json()["completion"]["text"], '\n')
        except KeyboardInterrupt:
            if does_model_exist():
                break
                delete_model()
    else:
        break
