from dataclasses import dataclass
import torch

@dataclass
class config:
    num_microbatches = 4
    batch_size = 2
    seq_len = 32
    d_model = 64
    dtype = torch.float32

class MainConfig:
    pp: config = config()

cfg = MainConfig()