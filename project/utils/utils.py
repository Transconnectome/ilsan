import os 
import glob 
from itertools import repeat
import collections.abc
import torch.nn as nn


def scaling_lr(batch_size, accumulation_steps, base_lr): 
    """
    Though the original code and paper additionally multiply n_gpus. 
    However, in this code, n_gpus is already included in batch size.
    Thus, scaling is done by multiplying batch_size * accumulation_steps * base_lr
    """
    # ref 1: https://github.com/CompVis/latent-diffusion/blob/main/main.py
    # ref 2: 
    return  batch_size * accumulation_steps * base_lr





# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)


def lambda_fn(module):
    if all(p.requires_grad for p in module.parameters()):
        return True  # wrap each trainable linear separately
    return False