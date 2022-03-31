import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import cupy


def torch2cupy(tensor):
    return cupy.fromDlpack(to_dlpack(tensor))


def cupy2torch(cupy_tensor):
    return from_dlpack(cupy_tensor.toDlpack())


def packbits(array):
    return cupy2torch(cupy.packbits(torch2cupy(array))), array.numel()


def unpackbits(array, size):
    return cupy2torch(cupy.unpackbits(torch2cupy(array))[:size])


def pack2bits(first, second):
    data = torch.cat((first, second.type(torch.bool)), 0)
    return cupy2torch(cupy.packbits(torch2cupy(data))), first.numel()


def unpack2bits(array, size):
    decode = cupy2torch(cupy.unpackbits(torch2cupy(array)))
    first = decode[:size]
    second = decode[size:2*size]
    second[first > 0] = 2

    return second


