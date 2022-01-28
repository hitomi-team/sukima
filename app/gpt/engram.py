import numpy as np
import torch

class Engrams:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def build(forward, tokens, shift=10000, factor=20000, rampdown=lambda x: x / 2):
        h = list(forward(input_ids=tokens[:, -512:].long().cuda(), output_hidden_states=True).hidden_states[1:])
        f = 0
        fa = 1.0 / float(len(h))

        for layer in range(len(h)):
            f = f + fa
            h[layer] = torch.mean(h[layer].detach().double(), dim=(1, )) * f

        h = torch.sum(torch.stack(h, axis=1)[0], dim=(0, ))
        return ((h + shift) / factor).float().to("cpu").numpy()
