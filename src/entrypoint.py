import os
import sys
import json

from server import Server

def entry():
    server = Server()
    server.run(host='0.0.0.0', port=8080)
    return 0

if __name__ == '__main__':
    entry()