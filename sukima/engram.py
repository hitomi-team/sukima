import os
import torch
import heapq
import numpy as np
import pickle

class Engrams:
    def __init__(self, memory_fp='memories.pkl', model=None, tokenizer=None):
        if os.path.exists(memory_fp):
            with open(memory_fp, 'rb') as f:
                self.memories = pickle.load(f)
        else:
            self.memories = []
        
        self.model = model
        self.tokenizer = tokenizer
    
    def build(forward, tokens, shift=10000, factor=20000, rampdown=lambda x:x / 2):
        h = list(forward(input_ids=tokens[:, -512:].long().cuda(), output_hidden_states=True).hidden_states[1:])
        f = 0
        fa = 1.0/float(len(h))

        for layer in range(len(h)):
            f = f + fa
            h[layer] = torch.mean(h[layer].detach().double(), dim=(1, )) * f
        
        h = torch.sum(torch.stack(h, axis=1)[0], dim=(0, ))
        return ((h + shift) / factor).float().to("cpu").numpy()
    
    def sort(self, now, past, factor=1000.0, epsilon=1e-6, top_k=250, depth=1, do_distance=True):
        now = now["engram"].astype(np.float32)

        if do_distance:
            for e in range(len(past)):
                past[e]["distance"] = np.sum(np.sqrt((np.abs(past[e]["engram"].astype(np.float32) - now) / factor) + epsilon))

        def keyer(m):
            if depth == 1:
                return m["distance"]
            else:
                total = 0
                nodeup = m
                nodedown = m

                # calculate distance across n previous and future engrams
                for e in range(depth-1):
                    nodeup = nodeup["previous"]
                    nodedown = nodedown["next"]
                    if nodeup is None or nodeup < 0 or nodedown is None or nodedown:
                        total = total + 100000
                        break
                    
                    f = (2.0 * (e + 1.0))

                    if nodeup < 0 or nodedown < 0:
                        total = total + 100000
                    else:
                        nodeup = past[nodeup]
                        nodedown = past[nodedown]
                        total = total + (nodeup["distance"] / f) + (nodedown["distance"] / f)
                return m["distance"] + total
        return heapq.nsmallest(top_k, past, key=keyer)


    def add(self, text):
        memory_count = len(self.memories)
        self.memories[-1]["next"] = memory_count
        engram = {
            "text": text,
            "engram": self.build(self.model.forward, self.tokenizer.encode(text, return_tensors="pt").input_ids),
            "next": -1,
            "previous": memory_count-1,
            "distance": 0
        }
        self.memories.append(engram)
        return engram
    
    def build_memories(self, now, short_term=10):
        # sort engrams
        m = self.sort(now, self.memories[:-short_term], top_k=600)
        m = self.sort(now, m, top_k=150, do_distance=False, depth=2)
        m = self.sort(now, m, top_k=42, do_distance=False, depth=3)
        m.reverse()
        return m

    def save(self, memory_fp='memories.pkl'):
        with open(memory_fp, 'wb') as f:
            pickle.dump(self.memories, f, protocol=pickle.HIGHEST_PROTOCOL)

# test
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    engrams = Engrams(model=model, tokenizer=tokenizer)
    engrams.add("haru: I like cheese!")
    engrams.add("soup6020: My favorite cheese is smoked cheddar.")

    m = engrams.build_memories(engrams.add("haru: What's your favorite cheese?"))
    text = ""
    for memory in m:
        text = text + memory["text"] + "\n"
    print(text)