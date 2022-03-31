from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import torch
import time
import random

has_gpu = True

class CompressorTest:
    """
    Tests for mergeComp compressor
    """

    def __init__(self, comp, mem, compress_ratio=0.01, ranks=1):
        self.set_current_context()
        self.compress_ratio = compress_ratio
        self.compressor = self.set_compressor(comp)
        self.memory = self.set_memory(mem)
        self.ranks = ranks
        self.residual = self.init_tensor()
        self.tensor = self.init_tensor()


    def set_current_context(self):
        if has_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.kwargs = {'dtype': torch.float32,
                'device': device,
                'requires_grad': False}


    def set_compressor(self, comp):
        sys.path.append("../")
        if comp == 'dgc':
            from mergeComp.compressor.pooldgc import PoolDgcCompressor
            return PoolDgcCompressor(compress_ratio=self.compress_ratio)
        elif comp == 'efsignsgd':
            from mergeComp.compressor.poolefsignsgd import PoolEFSignSGDCompressor
            return PoolEFSignSGDCompressor()
        elif comp == 'fp16':
            from mergeComp.compressor.poolfp16 import PoolFP16Compressor
            return PoolFP16Compressor()
        elif comp == 'none':
            from mergeComp.compressor.poolnone import PoolNoneCompressor
            return PoolNoneCompressor()
        elif comp == 'onebit':
            from mergeComp.compressor.poolonebit import PoolOneBitCompressor
            return PoolOneBitCompressor()
        elif comp == 'qsgd':
            from mergeComp.compressor.poolqsgd import PoolQSGDCompressor
            return PoolQSGDCompressor(quantum_num=127)
        elif comp == 'randomk':
            from mergeComp.compressor.poolrandomk import PoolRandomKCompressor
            return PoolRandomKCompressor(compress_ratio=self.compress_ratio)
        elif comp == 'signsgd':
            from mergeComp.compressor.poolsignsgd import PoolSignSGDCompressor
            return PoolSignSGDCompressor()
        elif comp == 'signum':
            from mergeComp.compressor.poolsignum import PoolSignumCompressor
            return PoolSignumCompressor(momentum=0.9)
        elif comp == 'terngrad':
            from mergeComp.compressor.poolterngrad import PoolTernGradCompressor
            return PoolTernGradCompressor()
        elif comp == 'topk':
            from mergeComp.compressor.pooltopk import PoolTopKCompressor
            return PoolTopKCompressor(compress_ratio=self.compress_ratio)
        else:
            raise NotImplementedError(comp)

        
    def set_memory(self, mem):
        sys.path.append("../")
        if mem == 'dgc':
            from mergeComp.memory.dgc import DgcMemory
            return DgcMemory()
        if mem == 'topk':
            from mergeComp.memory.topk import TopKMemory
            return TopKMemory()
        elif mem == 'none':
            from mergeComp.memory.none import NoneMemory
            return NoneMemory()
        elif mem == 'residual':
            from mergeComp.memory.residual import ResidualMemory
            return ResidualMemory()
        elif mem == 'efsignsgd':
            from mergeComp.memory.efsignsgd import EFSignSGDMemory
            return EFSignSGDMemory()
        else:
            raise NotImplementedError(mem)


    def init_tensor(self, size=2**26):
        return torch.rand(size, **self.kwargs)


    def compress(self, tensor):
        name = str(tensor.numel())
        tensor = self.memory.compensate(tensor, name)
        tensor_compressed, ctx = self.compressor.compress(tensor, name)
        self.memory.update(tensor, name, self.compressor, tensor_compressed, ctx)
        return tensor_compressed, ctx 


    def decompress(self, tensor, ctx):
        return self.compressor.decompress(tensor, ctx)


    def sum(self, tensor1, tensor2):
        return tensor1.add_(tensor2)

    
    def test_compressor(self, size, start=0):
        return self.compress(self.tensor[start:start+size])

        
    def test_decompressor(self, tensor, ctx):
        return self.decompress(tensor, ctx)


    def test_sum(self, size, start=0):
        return self.sum(self.tensor[start:start+size], self.tensor[start+size:start+2*size])


def test_compressor(comp, memory, result):
    compressor = CompressorTest(comp=comp, mem=memory)
    sizes = []
    for p in range(10, 26):
        size = 2 ** p
        sizes.append(size)
        torch.cuda.synchronize()
        compress_time, decompress_time, sum_time = 0, 0, 0
        skip = 5
        for i in range(runs):
            start = random.randint(0, 2**14)
            torch.cuda.synchronize()
            start_timestamp = time.time()
            tensor_compressed, ctx = compressor.test_compressor(size, start)
            torch.cuda.synchronize()
            compress_timestamp = time.time()
            if i >= skip:
                compress_time += compress_timestamp - start_timestamp
            compressor.test_decompressor(tensor_compressed, ctx)
            torch.cuda.synchronize()
            decompress_timestamp = time.time()
            if i >= skip:
                decompress_time += decompress_timestamp - compress_timestamp
            compressor.test_sum(size)
            torch.cuda.synchronize()
            if i >= skip:
                sum_time += time.time() - decompress_timestamp

        valid_runs = runs - skip
        result[0].append(round(compress_time/valid_runs*1000, 3))
        result[1].append(round(size*32 / (compress_time / valid_runs) / 1000 / 1000 / 1000, 3))
        result[2].append(round(decompress_time/valid_runs*1000, 3))
        result[3].append(round(size*32 / (decompress_time / valid_runs) / 1000 / 1000 / 1000, 3))
        result[4].append(round(sum_time/valid_runs*1000, 3))
        result[5].append(round(size*32 / (sum_time / valid_runs) / 1000 / 1000 / 1000, 3))
    print(comp, sizes, result, flush=True)

# the encoding+decoding overhead of FP16 is around 0.035ms
comp_list = [("fp16", "none"), ("efsignsgd", "efsignsgd"), ("dgc", "topk"), ("randomk", "topk"), ("onebit", "efsignsgd")]
runs = 20
results = {}

if __name__ == "__main__":
    for comp in comp_list:
        comp, memory = comp
        results[comp] = [[] for i in range(6)]
        test_compressor(comp, memory, results[comp])