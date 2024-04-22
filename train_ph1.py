"""
Reference code for Phi-1 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi/modeling_phi.py
"""

import torch
from torch import nn

class PhiRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(config.device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def forward(self, x):
        pass