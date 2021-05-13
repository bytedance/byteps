import torch
import byteps.torch as bps
import argparse
import os
import time

args = None

def get_arguments():
    global args
    parser = argparse.ArgumentParser(description='send recv benchmark')
    parser.add_argument('--size', type=int, default=1024,
                        help='number of elements in the tensor')
    parser.add_argument('--rank', type=int, default=None,
                        help='the rank')
    parser.add_argument('--test_mode', type=int, default=0,
                        help='0: rank 0 send once. '
                             '1: rank 0 sends twice. '
                             '2: rank 0 receives twice')
    parser.add_argument('--shuffle_rank', action='store_true',
                        help='shuffle rank assignment')
    parser.add_argument('--cpu', type=int, default=0,
                        help='use cpu context')

    args = parser.parse_args()

def receiver(rank):
    print('This is receiver with rank = {}'.format(args.rank))
    device = torch.device('cpu' if args.cpu else 'cuda:0') 
    tensor = torch.zeros(2, args.size, device=device)
    # send a tensor to myself
    print('calling recv_async (', rank, "->", args.rank, ") name = ", str(rank))
    handle = bps.recv_async(tensor, rank, name=str(rank))
    bps.synchronize(handle)
    print(tensor[0])
    print(tensor[1])
    print(f'Receiver is done on {device}')

def sender(rank):
    print('This is sender with rank = {}'.format(args.rank))
    device = torch.device('cpu' if args.cpu else 'cuda:0')
    tensor = torch.ones(2, args.size, device=device)
    # send a tensor to rank 1
    print('calling send_async (', args.rank, "->", rank, ") name = ", str(args.rank))
    handle = bps.send_async(tensor, rank, name=str(args.rank))
    bps.synchronize(handle)
    print(f'Sender is done on {device}')

def shuffle_rank(rank, total_workers):
    return (rank + 7) % total_workers

def main():
    get_arguments()
    total_workers = 2 if args.test_mode == 0 else 3
    my_rank = shuffle_rank(args.rank, total_workers) if args.shuffle_rank else args.rank
    bps.init(lazy=False, preferred_rank=my_rank)
    assert my_rank < total_workers
    if args.test_mode == 0:
        if my_rank == 0:
            sender(1)
        else:
            receiver(0)
    elif args.test_mode == 1:
        if my_rank == 0:
            sender(1)
            sender(2)
        else:
            receiver(0)
    elif args.test_mode == 2:
        if my_rank == 0:
            receiver(1)
            receiver(2)
        else:
            sender(0)
    else:
        raise ValueError("unexpected argument")
    bps.shutdown()

if __name__ == "__main__":
    main()