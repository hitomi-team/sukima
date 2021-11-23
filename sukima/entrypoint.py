import os
import sys
import json

import uvicorn

def entry():
    with open("config.json") as f:
        config = json.load(f)
    uvicorn.run("server:app", host=config["host"], port=config["port"], reload=True) # for debugging
    return 0

if __name__ == '__main__':
    sys.exit(entry())