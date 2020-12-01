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

import torch
import operator
from functools import reduce


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def get_type_size(dtype):
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float16 or dtype == torch.half or \
            dtype == torch.bfloat16:
        return 2
    elif dtype == torch.float64:
        return 8
    elif dtype == torch.int or dtype == torch.int32:
        return 4
    elif dtype == torch.int64 or dtype == torch.long:
        return 8
    elif dtype == torch.int16 or dtype == torch.short:
        return 2
    elif dtype == torch.int8:
        return 1
    else:
        raise ValueError("unknown dtype: %s" % dtype)


def size(tensor):
    return prod(tensor.shape) * get_type_size(tensor.dtype)


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
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class NagAdapter(Compressor):
    """For uncompressed gradients"""

    def __init__(self, compressor, mu, threshold, *args, **kwargs):
        self.compressor = compressor
        self.mu = mu
        self.mom = None
        self.threshold = threshold
        self.inited = False
        self.nag = False

    def compress(self, tensor, *args, **kwargs):
        """Returns the tensor unmodified."""
        return self.compressor.compress(tensor)

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Add nesterov momentum for uncompressed gradients"""
        tensor = self.compressor.decompress(tensor, ctx, *args, **kwargs)

        # uncompressed gradients need to do nag explicitly
        if not self.inited:
            if size(tensor.shape) < self.threshold:
                self.mom = torch.zeros_like(tensor)
                self.nag = True
            self.inited = True

        if self.nag:
            self.mom += tensor
            self.mom *= self.mu
            tensor += self.mom

        return tensor


class WeightDecayMomentumAdapter(Compressor):
    """For 1bit compression."""

    def __init__(self, compressor, mu, wd, threshold, *args, **kwargs):
        self.compressor = compressor
        self.mom = None
        self.cache = None
        self.mu = mu
        self.wd = wd
        self.threshold = threshold
        self.wdmom = False
        self.inited = False

    def compress(self, tensor, *args, **kwargs):
        """Returns the tensor unmodified."""
        return self.compressor.compress(tensor)

    def decompress(self, tensor, ctx, *args, **kwargs):
        """Returns the tensor added with additional momentum for wd
            m_t = \mu * m_{t-1} + wd * x_t
            x_{t+1} = x_t - \eta_t (tensor + \mu m_t + wd * x_t)
        """
        if "x" not in kwargs:
            raise ValueError("x is missing")

        x = kwargs["x"].type(tensor.dtype)

        if not self.inited:
            self.cache = torch.zeros_like(tensor)
            if size(tensor.shape) >= self.threshold:
                self.mom = torch.zeros_like(tensor)
                self.wdmom = True
            self.inited = True

        # weight decay
        self.cache = self.wd * x

        # weight decay momentum
        if self.wdmom:
            self.mom += self.cache
            self.mom *= self.mu
            tensor += self.mom

        tensor += self.cache
        return self.compressor.decompress(tensor, ctx, *args, **kwargs)


class Compression(object):
    """Optional gradient compression algorithm used during push_pull."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor()

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor()

    """Additional Momentum for weight decay. This is only for 1bit. This is a wrapper."""
    wdmom = WeightDecayMomentumAdapter

    """NAG for uncompressed. This is a wrapper."""
    nag = NagAdapter
