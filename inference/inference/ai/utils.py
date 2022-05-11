import torch
from transformers import AutoConfig

from inference.core.config import settings

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

from pathlib import Path

class Checkpoint(MutableMapping):
    def __init__(self, chkpt_dir, device="cpu"):
        self.device = device
        self.chkpt_dir = Path(chkpt_dir)
        self.checkpoint = torch.load(str(chkpt_dir / Path("m.pt")))

    def __len__(self):
        return len(self.checkpoint)

    def __getitem__(self, key):
        path = self.chkpt_dir / Path(self.checkpoint[key]).name

        if self.device == "cpu":
            return torch.load(str(path), map_location=self.device).long()
        else:
            return torch.load(str(path), map_location=self.device).half()

    def __setitem__(self, key, value):
        return

    def __delitem__(self, key, value):
        return

    def keys(self):
        return self.checkpoint.keys()

    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))

    def __copy__(self):
        return Checkpoint(self.chkpt_dir, device=self.device)

    def copy(self):
        return Checkpoint(self.chkpt_dir, device=self.device)

def get_dtype(device: torch.device):
    model_dtype = torch.float32
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_dtype = torch.float16
        else:
            device = torch.device('cpu')
            model_dtype = torch.float32
    else:
        if device == 'cuda':
            model_dtype = torch.float16
    return model_dtype

def is_decoder(config: AutoConfig):
    decoder_types = ['gpt2', 'gptj', 'gpt_neo', 'gpt_neox', 'xglm']
    encoder_types = ['distilbert', 'bert', 'xlm', 'xlm-roberta', 'roberta']

    if config.model_type in decoder_types:
        return True
    elif config.model_type in encoder_types:
        return False
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def tensorized_path(model_name: str):
    f = Path(settings.STORAGE_PATH) / Path(model_name.split('/')[-1])
    return f, f.with_suffix('.model').exists()
