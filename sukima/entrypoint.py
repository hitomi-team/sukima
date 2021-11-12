import os
import sys
import json

from server import Server

def entry():
    with open("config.json") as f:
        config = json.load(f)
    server = Server(config)
    server.run(host=config["host"], port=config["port"])
    return 0

if __name__ == '__main__':
    sys.exit(entry())