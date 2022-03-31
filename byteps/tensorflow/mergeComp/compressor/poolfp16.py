import tensorflow as tf
import sys
sys.path.append("../..")
from mergeComp import Compressor


class PoolFP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    def __init__(self):
        super().__init__()
        self.name = "PoolFP16"
        self.quantization = False

    def compress(self, tensor, name, start=0):
        """Downcasts the tensor to 16-bit."""
        dtype = tensor.dtype
        tensor_compressed = tensor
        if dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        ctx = (name, dtype)
        return [tensor_compressed], ctx


    def decompress(self, tensors, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_compressed = tensors[0]
        _, dtype = ctx
        tensor_decompressed = tensor_compressed
        #print("[decompress] before", ctx, torch.sum(tensor_compressed))
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor_compressed, dtype=dtype)
        #print("[decompress] after", ctx, torch.sum(tensor_compressed))
        return tensor_decompressed
