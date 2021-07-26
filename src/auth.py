import hashlib, secrets
import time
from tinydb import TinyDB, Query

class Auth():
    def __init__(self, config):
        self.config = config
        self.db = TinyDB(self.config["auth_db"])
        self.private_key = Query()
    
    def hash(self, key):
        return hashlib.sha256(key.encode('utf-8')).hexdigest()
    
    def auth_key(self, key):
        if not self.db.search(self.private_key.key == self.hash(key)):
            return False
        else:
            return True
    
    def create_key(self):
        key = "sk-" + secrets.token_hex(29)
        self.db.insert({"key": self.hash(key), "created-at": int(time.time())})
        return key
    
    def delete_key(self, key):
        key = Query()
        self.db.remove(key.key == self.hash(key))
