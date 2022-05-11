class AutoHF:
    def __init__(self, model_name='generic', decoder=False):
        self.model_name = model_name
        self.decoder = decoder

    def generate(self, args):
        raise NotImplementedError
    
    def classify(self, args):
        raise NotImplementedError

    def hidden(self, args):
        raise NotImplementedError
