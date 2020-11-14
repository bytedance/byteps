from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import timeit

import tensorflow as tf
import byteps.tensorflow.keras as bps
from tensorflow.keras import applications

tf.compat.v1.disable_eager_execution()

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda

bps.init()

# pin GPU to be used to process local rank (one GPU per process)
if args.cuda:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[bps.local_rank()], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

callbacks = [
    # BytePS: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    bps.callbacks.BroadcastGlobalVariablesCallback(0),
]
# Set up standard model.
model = getattr(applications, args.model)(weights=None)
opt = tf.keras.optimizers.Adam(0.01)
opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, loss_scale="dynamic")
opt = bps.DistributedOptimizer(opt)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy', 'top_k_categorical_accuracy'],
              experimental_run_tf_function=False)
model.fit(data, target, epochs=10, steps_per_epoch=16, callbacks=callbacks)

test_loss, test_acc, test_topk = model.evaluate(data, target, verbose=2, steps=16)
print('\nTest accuracy:', test_acc)
