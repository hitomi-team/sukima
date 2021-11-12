from flask import Flask, jsonify
from api import API
from util import Util

class Server:
    def __init__(self, config):
        self.app = Flask(__name__)
        self.config = config
        self.api = API(version='v1', app=self.app, config=config)
    
    def page_not_found(self, e):
        return jsonify(Util.error(404, "Invalid URL"))
    
    def run(self, host, port):
        self.api.run()
        self.app.register_error_handler(404, self.page_not_found)
        self.app.run(host=host, port=port, ssl_context='adhoc')