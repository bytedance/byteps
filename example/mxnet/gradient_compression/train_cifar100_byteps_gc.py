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
"""This file is modified from
`gluon-cv/scripts/classification/cifar/train_cifar10.py`"""
import argparse
import logging
import subprocess
import time

import gluoncv as gcv
import matplotlib
import mxnet as mx
from gluoncv.data import transforms as gcv_transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils import LRScheduler, LRSequential, makedirs
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import byteps.mxnet as bps

matplotlib.use('Agg')


gcv.utils.check_version('0.6.0')


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='resnet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='weight decay rate. default is 0.0005.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 100,150.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--logging-file', type=str, default='baseline',
                        help='name of training log file')
    # additional arguments for gradient compression
    parser.add_argument('--compressor', type=str, default='',
                        help='which compressor')
    parser.add_argument('--ef', type=str, default='',
                        help='which error-feedback')
    parser.add_argument('--compress-momentum', type=str, default='',
                        help='which compress momentum')
    parser.add_argument('--onebit-scaling', action='store_true', default=False,
                        help='enable scaling for onebit compressor')
    parser.add_argument('--k', default=1.0, type=float,
                        help='topk or randomk')
    parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                        help='use fp16 compression during pushpull')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    bps.init()

    gpu_name = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
    gpu_name = gpu_name.decode('utf8').split('\n')[-2]
    gpu_name = '-'.join(gpu_name.split())
    filename = "cifar100-%d-%s-%s.log" % (bps.size(),
                                          gpu_name, opt.logging_file)
    filehandler = logging.FileHandler(filename)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 100

    num_gpus = opt.num_gpus
    # batch_size *= max(1, num_gpus)
    context = mx.gpu(bps.local_rank()) if num_gpus > 0 else mx.cpu(
        bps.local_rank())
    num_workers = opt.num_workers
    nworker = bps.size()
    rank = bps.rank()

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]

    num_batches = 50000 // (opt.batch_size * nworker)
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=opt.warmup_lr,
                    target_lr=opt.lr * nworker / bps.local_size(),
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('step', base_lr=opt.lr * nworker / bps.local_size(),
                    target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    num_batches = 50000 // (opt.batch_size * nworker)
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=opt.warmup_lr, target_lr=opt.lr * nworker / bps.local_size(),
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('step', base_lr=opt.lr * nworker / bps.local_size(), target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                  'drop_rate': opt.drop_rate}
    else:
        kwargs = {'classes': classes}
    net = get_model(model_name, **kwargs)
    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)

    if opt.compressor:
        optimizer = 'sgd'
    else:
        optimizer = 'nag'

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    # from https://github.com/weiaicunzai/pytorch-cifar/blob/master/conf/global_settings.py
    CIFAR100_TRAIN_MEAN = [0.5070751592371323,
                           0.48654887331495095, 0.4409178433670343]
    CIFAR100_TRAIN_STD = [0.2673342858792401,
                          0.2564384629170883, 0.27615047132568404]

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN,
                             CIFAR100_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN,
                             CIFAR100_TRAIN_STD)
    ])

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=True).shard(
                nworker, rank).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard',
            num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=False).shard(
                nworker, rank).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        params = net.collect_params()

        compression_params = {
            "compressor": opt.compressor,
            "ef": opt.ef,
            "momentum": opt.compress_momentum,
            "scaling": opt.onebit_scaling,
            "k": opt.k,
            "fp16": opt.fp16_pushpull
        }

        optimizer_params = {'lr_scheduler': lr_scheduler,
                            'wd': opt.wd, 'momentum': opt.momentum}

        trainer = bps.DistributedTrainer(params,
                                         optimizer,
                                         optimizer_params,
                                         compression_params=compression_params)
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        iteration = 0
        best_val_score = 0
        bps.byteps_declare_tensor("acc")
        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(
                    batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(
                    batch[1], ctx_list=ctx, batch_axis=0)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                train_metric.update(label, output)
                name, train_acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, train_acc = train_metric.get()
            throughput = int(batch_size * nworker * i / (time.time() - tic))

            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f lr=%f' %
                        (epoch, throughput, time.time()-tic, trainer.learning_rate))

            name, val_acc = test(ctx, val_data)
            acc = mx.nd.array([train_acc, val_acc], ctx=ctx[0])
            bps.byteps_push_pull(acc, name="acc", is_average=False)
            acc /= bps.size()
            train_acc, val_acc = acc[0].asscalar(), acc[1].asscalar()
            if bps.rank() == 0:
                logger.info('[Epoch %d] training: %s=%f' %
                            (epoch, name, train_acc))
                logger.info('[Epoch %d] validation: %s=%f' %
                            (epoch, name, val_acc))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-cifar-%s-%d-best.params' %
                                    (save_dir, best_val_score, model_name,
                                     epoch))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters('%s/cifar100-%s-%d.params' %
                                    (save_dir, model_name, epoch))

        if save_period and save_dir:
            net.save_parameters('%s/cifar100-%s-%d.params' %
                                (save_dir, model_name, epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)


if __name__ == '__main__':
    main()
