import torch
import time
from mergeComp_dl.torch.compressor.poolsignsgd import PoolSignSGDCompressor
from mergeComp_dl.torch.compressor.pooldgc import PoolDgcCompressor
from mergeComp_dl.torch.compressor.pooltopk import PoolTopKCompressor
from mergeComp_dl.torch.compressor.poolfp16 import PoolFP16Compressor

#compressor = PoolSignSGDCompressor()
#compressor = PoolDgcCompressor(0.01)
#compressor = PoolTopKCompressor(0.01)
compressor = PoolFP16Compressor()

base_size = 2 ** 10
device = torch.device("cpu")

kwargs = {'dtype': torch.float32,
          'device': device,
          'requires_grad': False}

name = "test"
size_list = []
compress_latency = []
decompress_latency = []

runs = 20

for i in range(0, 18, 1):
    ctx = None 
    size = base_size * 2 ** i
    size_list.append(10 + i)
    compress_time, decompress_time = 0, 0

    for _ in range(0, runs):
        tensor = torch.rand(size, **kwargs)
        torch.cuda.synchronize()
        start_time = time.time()
        compressed_tensor, ctx = compressor.compress(tensor, name, ctx)
        torch.cuda.synchronize()
        end_time = time.time()
        
        compress_time += end_time-start_time
        
        #print("Compress, size:", size, "time:", end_time-start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        decompressed = compressor.decompress(compressed_tensor, ctx)
        torch.cuda.synchronize()
        end_time = time.time()
        decompress_time += end_time-start_time

        #print("Decompress, size:", size, "time:", end_time-start_time)

    compress_latency.append(round(compress_time*1000/runs, 2)) 
    decompress_latency.append(round(decompress_time*1000/runs, 2))

print(size_list)
print(compress_latency)
print(decompress_latency)