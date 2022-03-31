import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import numpy as np


def packbits(array):
    return np.packbits(array)


def unpackbits(array, size):
    return torch.from_numpy(np.unpackbits(array)[:size])
