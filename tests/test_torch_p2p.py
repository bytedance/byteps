import numpy as np
import time, os
import torch

import argparse
parser = argparse.ArgumentParser(description='PyTorch tests')
parser.add_argument('--rank', type=int, default=-1)
parser.add_argument('--backend', type=str, default='byteps')

args = parser.parse_args()
if args.backend == 'byteps':
    import byteps.torch as bps
    bps.init(lazy=False, preferred_rank=args.rank)
else:
    import horovod.torch as bps
    bps.init()

print(bps)

class TorchTests:
    """
    Tests for ops in byteps.torch.
    """
    def __init__(self):
        self.rank = bps.rank()
        self.size = bps.size()

    def test_send_recv(self):
        def receiver(rank):
            print('This is receiver with rank = {}'.format(self.rank), flush=True)
            device = torch.device('cuda')
            tensor = torch.zeros(self.size, device=device)
            handle = bps.recv_async(tensor, rank, name=str(rank))
            bps.synchronize(handle)
            print(tensor, flush=True)
            print('Receiver is done', flush=True)

        def sender(rank):
            print('This is sender with rank = {}'.format(self.rank), flush=True)
            device = torch.device('cuda')
            tensor = torch.ones(self.size, device=device)
            handle = bps.send_async(tensor, rank, name=str(self.rank))
            bps.synchronize(handle)
            print('Sender is done')

        if self.rank == 0:
            sender(1)
        elif self.rank == 1:
            receiver(0)
        else:
            raise ValueError("unexpected rank")
        print('test_send_recv DONE', flush=True)

tests = TorchTests()
# TODO test send to myself
# tests.test_self_send_recv()
tests.test_send_recv()
time.sleep(3)