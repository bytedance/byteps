# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This file is modified from `horovod/examples/mxnet_mnist.py`, using gluon style MNIST dataset and data_loader."""
import time

import argparse
import logging

import mxnet as mx
import byteps.mxnet as bps
from mxnet import autograd, gluon, nd
from mxnet.gluon.data.vision import MNIST


# Higher download speed for chinese users
# os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

# Training settings
parser = argparse.ArgumentParser(description='MXNet MNIST Example')

parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--j', type=int, default=2,
                    help='number of cpu processes for dataloader')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable training on GPU (default: False)')
args = parser.parse_args()

if not args.no_cuda:
    # Disable CUDA if there are no GPUs.
    if mx.context.num_gpus() == 0:
        args.no_cuda = True

logging.basicConfig(level=logging.INFO)
logging.info(args)


def dummy_transform(data, label):
    im = data.astype(args.dtype, copy=False) / 255 - 0.5
    im = nd.transpose(im, (2, 0, 1))
    return im, label


# Function to get mnist iterator
def get_mnist_iterator():
    train_set = MNIST(train=True, transform=dummy_transform)
    train_iter = gluon.data.DataLoader(train_set, args.batch_size, True, num_workers=args.j, last_batch='discard')
    val_set = MNIST(train=False, transform=dummy_transform)
    val_iter = gluon.data.DataLoader(val_set, args.batch_size, False, num_workers=args.j)

    return train_iter, val_iter, len(train_set)


# Function to define neural network
def conv_nets():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(10))
    return net


# Function to evaluate accuracy for a model
def evaluate(model, data_iter, context):
    metric = mx.metric.Accuracy()
    for _, batch in enumerate(data_iter):
        data = batch[0].as_in_context(context)
        label = batch[1].as_in_context(context)
        output = model(data.astype(args.dtype, copy=False))
        metric.update([label], [output])

    return metric.get()


# Load training and validation data
train_data, val_data, train_size = get_mnist_iterator()

# Initialize BytePS
bps.init()

# BytePS: pin context to local rank
context = mx.cpu(bps.local_rank()) if args.no_cuda else mx.gpu(bps.local_rank())
num_workers = bps.size()

# Build model
model = conv_nets()
model.cast(args.dtype)

# Initialize parameters
model.initialize(mx.init.MSRAPrelu(), ctx=context)
# if bps.rank() == 0:
model.summary(nd.ones((1, 1, 28, 28), ctx=mx.gpu(bps.local_rank())))
model.hybridize()

params = model.collect_params()

# BytePS: create DistributedTrainer, a subclass of gluon.Trainer
optimizer_params = {'momentum': args.momentum, 'learning_rate': args.lr * num_workers}
trainer = bps.DistributedTrainer(params, "sgd", optimizer_params)

# Create loss function and train metric
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()

# Train model
for epoch in range(args.epochs):
    tic = time.time()
    metric.reset()
    for i, batch in enumerate(train_data):
        data = batch[0].as_in_context(context)
        label = batch[1].as_in_context(context)

        with autograd.record():
            output = model(data)
            loss = loss_fn(output, label)

        loss.backward()
        trainer.step(args.batch_size)
        metric.update([label], [output])

        if i % 100 == 0:
            name, acc = metric.get()
            logging.info('[Epoch %d Batch %d] Training: %s=%f' %
                         (epoch, i, name, acc))

    if bps.rank() == 0:
        elapsed = time.time() - tic
        speed = train_size * num_workers / elapsed
        logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
                     epoch, speed, elapsed)

    # Evaluate model accuracy
    _, train_acc = metric.get()
    name, val_acc = evaluate(model, val_data, context)
    if bps.rank() == 0:
        logging.info('Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f', epoch, name,
                     train_acc, name, val_acc)

    if bps.rank() == 0 and epoch == args.epochs - 1:
        assert val_acc > 0.96, "Achieved accuracy (%f) is lower than expected\
                                (0.96)" % val_acc
