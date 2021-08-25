#!/usr/bin/python3

import argparse
import os
import numpy as np
import time

import tensorflow as tf
import byteps.tensorflow as bps

# tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args, unknown = parser.parse_known_args()
args.cuda = False


bps.init(lazy=False)
my_rank = bps.rank()

if args.cuda:
    print("using cuda")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[bps.local_rank()], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@tf.function
def run_test(grads):
    grads = [bps.push_pull(grad) for grad in grads]
    return grads

print("my_rank ", my_rank)
tensor_size = 1024000
test_dtype = tf.float32
with tf.device("CPU"):
  grads = [tf.ones([tensor_size, 1], dtype = test_dtype) * my_rank for i in range(1)]
for ii in range(2):
  ret = run_test(grads)

  for idx, grad in enumerate(ret):
      with tf.device("CPU"):
        expected_tensor = tf.ones([tensor_size, 1]) * (bps.size() * (bps.size() - 1) / 2 / bps.size())
        is_equal = tf.equal(expected_tensor, tf.cast(grad, dtype=tf.float32))
      if tf.reduce_all(is_equal):
          tf.print(f"rank {my_rank} sum #{idx} OK ")
      else:
          tf.print("rank {} sum #{} wrong".format(my_rank, idx))
          tf.print("rank ", my_rank, "after: \n", grad)
      tf.print("Round ", ii, " done")
