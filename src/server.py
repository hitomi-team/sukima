from gptneo import GPTNeo

from flask import Flask, request, jsonify
from api import API
from util import Util

class Server:
    def __init__(self):
        self.app = Flask(__name__)
        self.api = API(version='v1', app=self.app)
    
    def page_not_found(self, e):
        return jsonify(Util.error(404, "Invalid URL"))
    
    def run(self, host, port):
        self.api.run()
        self.app.register_error_handler(404, self.page_not_found)
        self.app.run(host=host, port=port)