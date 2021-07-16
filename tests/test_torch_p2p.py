import numpy as np
import time, os
import torch

import argparse
parser = argparse.ArgumentParser(description='PyTorch tests')
args = parser.parse_args()

import byteps.torch as bps
bps.init()

class TorchTests:
    """
    Tests for ops in byteps.torch.
    """
    def __init__(self):
        self.rank = bps.rank()
        self.size = bps.size()

    def test_send_recv(self):
        def receiver(from_rank, to_rank, device):
            print('This is receiver with rank = {}'.format(self.rank), flush=True)
            device = torch.device(device)
            expected = torch.ones(self.size).numpy()
            tensor = torch.zeros(self.size, device=device)
            handle = bps.recv_async(tensor, from_rank, name=f"test_{device}")
            bps.synchronize(handle)
            assert np.sum(tensor != expected), (result, expected)
            print('Receiver is done', flush=True)

        def sender(from_rank, to_rank, device):
            print('This is sender with rank = {}'.format(self.rank), flush=True)
            device = torch.device(device)
            tensor = torch.ones(self.size, device=device)
            handle = bps.send_async(tensor, to_rank, name=f"test_{device}")
            bps.synchronize(handle)
            print('Sender is done')

        if self.rank == 0:
            sender(0, 1, 'cuda')
            sender(0, 1, 'cpu')
        elif self.rank == 1:
            receiver(0, 1, 'cuda')
            receiver(0, 1, 'cpu')
        else:
            raise ValueError("unexpected rank")
        print('test_send_recv DONE', flush=True)

tests = TorchTests()
tests.test_send_recv()
time.sleep(3)