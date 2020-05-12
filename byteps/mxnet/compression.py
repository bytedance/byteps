# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""
import mxnet
import mxnet.ndarray as nd

import threading
from queue import Queue, Empty


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    def compress(self, tensor, *args, **kwargs):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, *args, **kwargs):
        """Returns the tensor unmodified."""
        return tensor, None

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    def compress(self, tensor, *args, **kwargs):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if 'float' in str(self, tensor.dtype):
            # Only allow compression from other floating point types
            tensor_compressed = tensor.astype('float16', copy=False)
        return tensor_compressed, tensor.dtype

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if 'float' in str(dtype):
            tensor_decompressed = tensor.astype(dtype, copy=False)
        return tensor_decompressed


class WeightDecayMomentum(Compressor):
    """For 1bit compression."""

    def __init__(self, compressor, mu, wd, *args, **kwargs):
        self.compressor = compressor
        self.mu = mu
        self.wd = wd
        self.task_queue = Queue()
        self.done_queue = Queue()
        threading.Thread(target=self._worker, args=(
            self.mu, self.wd, self.task_queue, self.done_queue), daemon=True).start()

    def __del__(self):
        self.task_queue.put('STOP')

    @staticmethod
    def _worker(mu, wd, input, output):
        mom = None
        cache = None
        for x, _ in iter(input.get, 'STOP'):
            if mom is None:
                mom = nd.zeros_like(x)
                cache = nd.zeros_like(x)

            nd._internal._mul_scalar(x, wd, out=cache)
            mom += cache
            nd._internal._mul_scalar(mom, mu, out=mom)
            cache += mom
            output.put(cache)

    def compress(self, tensor, *args, **kwargs):
        """Returns the tensor unmodified."""
        if "x" not in kwargs:
            return self.compressor.compress(tensor)

        x = kwargs["x"]
        self.task_queue.put((x, None))
        return self.compressor.compress(tensor)

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Returns the tensor added with additional momentum for wd
            m_t = \mu * m_{t-1} + wd * x_t
            x_{t+1} = x_t - \eta_t (tensor + \mu m_t + wd * x_t)
        """
        try:
            tensor += self.done_queue.get(timeout=0.1)
        except Empty:
            print("empty for wd-momentum")
        except TimeoutError:
            print("timeout for wd-momentum")
        return self.compressor.decompress(tensor, ctx)


class Compression(object):
    """Optional gradient compression algorithm used during push_pull."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor()

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor()

    """Additional Momentum for weight decay. This is only for 1bit. This is a wrapper."""
    wdmom = WeightDecayMomentum


# if __name__ == "__main__":
#     x = WeightDecayMomentum(Compression.none, 0.9, 1e-4)
#     import copy
#     print(x.__dict__)
#     y = type(x)(**x.__dict__)
#     print(y.__dict__)
#     print(x.task_queue is y.task_queue)
