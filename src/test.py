import requests

request_body = {
    "model": "gpt-neo-125M",
    "prompt": "The touhou project",
    "generate_num": 32,
    "temperature": 0.1,
    "top_p": 1.0,
    "repetition_penalty": 3.0
}

while True:
    res = requests.post('http://0.0.0.0:8080/v1/generate', json=request_body)
    print(res.json())
