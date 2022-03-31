import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import cupy

zero_paddings = torch.zeros((32), dtype=torch.bool).cuda()

def torch2cupy(tensor):
    return cupy.fromDlpack(to_dlpack(tensor))


def cupy2torch(cupy_tensor):
    return from_dlpack(cupy_tensor.toDlpack())


def packbits(array, unit_size=8):
    tensor = array
    numel = tensor.numel()
    if numel % unit_size != 0:
        padding_size = unit_size - numel % unit_size
        tensor = torch.cat((tensor, zero_paddings[:padding_size]), dim=0)
    
    if unit_size == 8:
        return cupy2torch(cupy.packbits(torch2cupy(array)))
    elif unit_size == 16:
        cupy_tensor = cupy.packbits(torch2cupy(tensor))
        return cupy2torch(cupy_tensor.view(cupy.float16))
    elif unit_size == 32:
        cupy_tensor = cupy.packbits(torch2cupy(tensor))
        return cupy2torch(cupy_tensor.view(cupy.float32))
    else:
        raise AttributeError("unsupported data type size")


def unpackbits(array, size):
    return cupy2torch(cupy.unpackbits(torch2cupy(array).view(cupy.uint8))[:size])


def pack2bits(first, second):
    data = torch.cat((first, second.type(torch.bool)), 0)
    return cupy2torch(cupy.packbits(torch2cupy(data)))


def unpack2bits(array, size):
    decode = cupy2torch(cupy.unpackbits(torch2cupy(array)))
    first = decode[:size]
    second = decode[size:2*size]
    second[first > 0] = 2

    return second