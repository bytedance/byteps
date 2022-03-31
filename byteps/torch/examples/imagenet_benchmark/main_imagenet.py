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
import functools
import io, os
import subprocess
import sys
import torch
import torch.utils.data as data
import multiprocessing as mp

try:
    from dataloader import KVReader
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'byted-dataloader==0.2.6', "-i", "https://bytedpypi.byted.org/simple"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pillow'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'torchvision'])
    from dataloader import KVReader
    from PIL import Image

sys.path.append("../../")
from mergeComp.helper import add_parser_arguments, wrap_compress_optimizer, init_comm


# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='vgg16',
                    help='model to benchmark')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--batches-per-pushpull', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing pushpull across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=256,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')

# https://github.com/pytorch/examples/tree/master/imagenet
parser.add_argument('--base-lr', type=float, default=0.00125,
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

init_comm(args.local_rank)
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    # BytePS: pin GPU to local rank.
    print("rank:", bps.local_rank())
    torch.cuda.set_device(bps.local_rank())
cudnn.benchmark = True

CN_DATASET_PATHS = {"train": "hdfs://haruna/home/byte_arnold_lq/user/data/imagenet1k/train",
                    "val": "hdfs://haruna/home/byte_arnold_lq/user/data/imagenet1k/val"}
US_DATASET_PATHS = {"train": "hdfs://harunava/home/byte_arnold_va_mlsys/data/imagenet1k/train",
                    "val": "hdfs://harunava/home/byte_arnold_va_mlsys/data/imagenet1k/val"}
PATH_MAP = {"cn":CN_DATASET_PATHS,
            "us": US_DATASET_PATHS}

def get_keys(args):
    return KVReader(*args).list_keys()

class ImagenetDataset(data.Dataset):
    def __init__(self, mode="train", num_readers=2, loc="us", preprocessor=None):
        self.path = PATH_MAP[loc][mode]
        self.num_readers = num_readers
        self.preprocessor = preprocessor
        with mp.Pool(1) as p:
            self.keys = p.map(get_keys, [(self.path, num_readers)])[0]

    def init_reader(self, path, num_readers):
        self.reader = KVReader(path, num_readers)
        classes = list(set([key.split("/n")[ -1] .split("_")[0] for key in self.keys]))
        classes.sort()
        self.class_str_to_index = {clss: idx for idx, clss in enumerate(classes)}

    def __len__(self):
        #return len(KVReader(self.path, 1).list_keys())
        return len(self.keys)
    def __getitem__(self, index):
        if not hasattr(self, "reader"):
            self.init_reader(self.path, self.num_readers)
        if isinstance(index, list):
            if len(index) == 0:
                return []
            keys = [self.keys[i] for i in index]
            imgs = [Image.open(io.BytesIO(img_bin)).convert('RGB') for img_bin in self.reader.read_many(keys)]
            labels = [self.class_str_to_index[label.split("/n")[ -1] .split("_")[0]] for label in keys]
            return zip(imgs, labels)
        elif isinstance(index, int):
            key = self.keys[index]
            img_bin = self.reader.read_many([key])[0]
            img = Image.open(io.BytesIO(img_bin)).convert('RGB')
            label = self.class_str_to_index[key.split("/n")[ -1] .split("_")[0]]
            if self.preprocessor:
                return self.preprocessor.preprocess(data=(img, label))
            return (img, label)

        else:
            raise LookupError('Unsupported index: list and int are supported')

    #@property
    #@functools.lru_cache()
    #def keys(self):
    #    return self.reader.list_keys()


class ImagenetDatasetPreprocessor:
    def __init__(self):

        self.torch_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    def transform(self, data):
        img, label = data
        # need to return a boolean to indicate if the transformed data is valid
        return (self.torch_transforms(img), label), True

    def preprocess(self, data):
        img, label = data
        return (self.torch_transforms(img), label)

    def batch_transform(self, data):
        images, labels = [], []
        for item in data:
            images.append(item[0])
            labels.append(item[1])
        return torch.stack(images), torch.tensor(labels)


kwargs = {'num_workers': 4, 'pin_memory': True} 
train_dataset = ImagenetDataset(mode="train", preprocessor=ImagenetDatasetPreprocessor(), num_readers=2)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bps.size(), rank=bps.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    **kwargs)

val_dataset = ImagenetDataset(mode="val", preprocessor=ImagenetDatasetPreprocessor(), num_readers=2)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bps.size(), rank=bps.rank())
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.val_batch_size,
    sampler=val_sampler,
    **kwargs)


# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
# for try_epoch in range(args.epochs, 0, -1):
#     if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
#         resume_from_epoch = try_epoch
#         break

# BytePS: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
#resume_from_epoch = bps.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
#                                  name='resume_from_epoch').item()

# BytePS: print logs on the first worker.
verbose = 1 if bps.rank() == 0 else 0

# BytePS: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if bps.rank() < 0 else None

model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# BytePS: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_pushpull
optimizer = optim.SGD(model.parameters(),
                      lr=args.base_lr * bps.size(),
                      momentum=args.momentum, weight_decay=args.wd)

# BytePS: broadcast parameters & optimizer state.
# bps.broadcast_parameters(model.state_dict(), root_rank=0)
# bps.broadcast_optimizer_state(optimizer, root_rank=0)

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
        for batch_idx, (data, target) in enumerate(train_loader):
            # the first few iterations are unstable
            adjust_learning_rate(epoch, batch_idx)

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

            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


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
    save_checkpoint(epoch)

sleep(1)
os.system("pkill -9 python3")