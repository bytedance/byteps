import logging
import time
import argparse
import byteps.mxnet as bps
from gluoncv.data import transforms as gcv_transforms
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.model_zoo import get_model
import gluoncv as gcv
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import gluon, nd
import mxnet as mx
import numpy as np
import matplotlib
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
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
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
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    parser.add_argument('--logging-file', type=str, default='train_cifar100.log',
                        help='name of training log file')
    # additional arguments for gradient compression
    parser.add_argument('--compressor', type=str, default='',
                        help='which compressor')
    parser.add_argument('--ef', type=str, default=None,
                        help='enable error-feedback')
    parser.add_argument('--onebit-scaling', action='store_true', default=False,
                        help='enable scaling for onebit compressor')
    parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                        help='use fp16 compression during pushpull')
    parser.add_argument('--compress-momentum', action='store_true', default=False,
                        help='enable compress momentum.')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    filehandler = logging.FileHandler(opt.logging_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    bps.init()

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
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                  'drop_rate': opt.drop_rate}
    else:
        kwargs = {'classes': classes}
    net = get_model(model_name, **kwargs)
    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)
    optimizer = 'sgd'

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    plot_path = opt.save_plot_dir

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
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

        train_dataset = gluon.data.vision.CIFAR100(train=True)
        length = len(train_dataset)
        begin = length * int(float(rank) / nworker)
        end = length * int(float(rank+1) / nworker)
        train_data = gluon.data.DataLoader(
            train_dataset[begin:end].transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_dataset = gluon.data.vision.CIFAR100(train=False)
        length = len(val_dataset)
        begin = length * int(float(rank) / nworker)
        end = length * int(float(rank+1) / nworker)
        val_data = gluon.data.DataLoader(
            val_dataset[begin:end].transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        params = net.collect_params()
        if opt.compressor:
            for _, param in params.items():
                setattr(param, "byteps_compressor_type", opt.compressor)
                if opt.ef:
                    setattr(param, "byteps_error_feedback_type", opt.ef)
                if opt.onebit_scaling:
                    setattr(
                        param, "byteps_compressor_onebit_enable_scale", opt.onebit_scaling)
                if opt.compress_momentum:
                    setattr(param, "byteps_momentum_type", "vanilla")
                    setattr(param, "byteps_momentum_mu", opt.momentum)

        optimizer_params = {'learning_rate': opt.lr *
                            nworker, 'wd': opt.wd, 'momentum': opt.momentum}
        if opt.compress_momentum:
            del optimizer_params["momentum"]

        compression = bps.Compression.fp16 if opt.fp16_pushpull else bps.Compression.none
        trainer = bps.DistributedOptimizer(params,
                                           optimizer, optimizer_params, compression=compression)
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

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
                name, acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            throughput = int(batch_size * nworker * i / (time.time() - tic))

            if rank == 0:
                logger.info('[Epoch %d] training: %s=%f' %
                            (epoch, name, acc))
                logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' %
                            (epoch, throughput, time.time()-tic))

            name, val_acc = test(ctx, val_data)
            if rank == 0:
                logger.info('[Epoch %d] validation: %s=%f' %
                            (epoch, name, val_acc))

            train_history.update([1-acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png' %
                               (plot_path, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-cifar-%s-%d-best.params' %
                                    (save_dir, best_val_score, model_name, epoch))

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
