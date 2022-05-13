from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import byteps.torch as bps
import tensorboardX
import sys, os
import math
from tqdm import tqdm
from time import time as time_
from time import sleep

sys.path.append("../../")
from mergeComp.helper import add_parser_arguments, wrap_compress_optimizer, init_comm


# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='vgg16',
                    help='model to benchmark')
parser.add_argument('--train-dir', default=os.path.expanduser('~/data/imagewang/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/data/imagewang/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--batches-per-pushpull', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing pushpull across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=64,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--speed_test', action='store_true', default=False,
                    help='random seed')
parser = add_parser_arguments(parser)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

pushpull_batch_size = args.batch_size * args.batches_per_pushpull

init_comm(args.local_rank)
torch.manual_seed(args.seed)

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# BytePS: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
#resume_from_epoch = bps.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
#                                  name='resume_from_epoch').item()

# BytePS: print logs on the first worker.
verbose = 1 if bps.rank() == 0 else 0

# BytePS: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if bps.rank() < 0 else None


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# BytePS: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=bps.size()` and `rank=bps.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bps.size(), rank=bps.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=pushpull_batch_size,
    sampler=train_sampler, **kwargs)

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bps.size(), rank=bps.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)

model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# BytePS: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_pushpull
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_pushpull * bps.size()),
                      momentum=args.momentum, weight_decay=args.wd)

# BytePS: broadcast parameters & optimizer state.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

optimizer = wrap_compress_optimizer(model, optimizer, args)

# Restore from a previous checkpoint, if initial_epoch is specified.
# BytePS: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and bps.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        start_time, end_time= 0, 0
        start_idx = int(len(train_loader) * 0.25)
        end_idx = int(len(train_loader) * 0.75)
        for batch_idx, (data, target) in enumerate(train_loader):
            # the first few iterations are unstable
            if batch_idx == start_idx:
                torch.cuda.synchronize()
                start_time = time_()
            adjust_learning_rate(epoch, batch_idx)

            if args.profile and end_idx >= batch_idx >= start_idx:
                torch.cuda.synchronize()
                profile_start = time_() 

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            
            if args.profile and end_idx >= batch_idx >= start_idx:
                torch.cuda.synchronize()
                profile_end = time_() 
                if bps.local_rank() == 0:
                    print("FP+BP time:", profile_end-profile_start)

            # Gradient is applied across all ranks
            optimizer.step()
            if batch_idx == end_idx:
                torch.cuda.synchronize()
                end_time = time_()

            if args.profile and end_idx >= batch_idx >= start_idx:
                torch.cuda.synchronize()
                if bps.local_rank() == 0:
                    print("step time:", time_() - profile_start)

            # if args.speed_test:
            #     continue
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)


    if args.speed_test and bps.local_rank() == 0:
        print("Epoch: {}\tIterations: {}\tTime: {:.2f} s\tTraining speed: {:.1f} img/s".format(
            epoch,
            len(train_loader),
            (end_time - start_time)/(end_idx - start_idx + 1),
            (end_idx - start_idx + 1)*args.batch_size/(end_time - start_time)*bps.size()
        ))

    # if log_writer:
    #     log_writer.add_scalar('train/loss', train_loss.avg, epoch)
    #     log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# BytePS: using `lr = base_lr * bps.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * bps.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / bps.size() * (epoch * (bps.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * bps.size() * args.batches_per_pushpull * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()


def save_checkpoint(epoch):
    if bps.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# BytePS: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        val = val.detach().cpu()
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
    # save_checkpoint(epoch)

sleep(1)
os.system("pkill -9 python3")