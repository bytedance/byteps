import torch
import byteps.torch as bps
import argparse
import os
import time

args = None

def get_arguments():
    global args
    parser = argparse.ArgumentParser(description='send recv benchmark')
    parser.add_argument('--size', type=int, default=1024000,
                        help='number of elements in the tensor')
    parser.add_argument('--niter', type=int, default=1,
                        help='number of iterations')
    parser.add_argument('--rank', type=int, default=None,
                        help='the rank')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='use CPU context')
    parser.add_argument('--busy_wait', action='store_true', default=False,
                        help='busy waiting durating sync')

    args = parser.parse_args()

def receiver():
    print('This is receiver with rank = {}'.format(args.rank))
    device = torch.device('cpu') if args.cpu else torch.device('cuda:0')
    tensor = torch.zeros(args.size, device=device)
    # send a tensor to myself
    print('calling recv_async')
    handle = bps.recv_async(tensor, 0, name='test')
    bps.synchronize(handle)
    print(tensor)
    total_bw = 0
    for i in range(args.niter):
        start_time = time.time()
        handle = bps.recv_async(tensor, 0, name='test')
        bps.synchronize(handle, args.busy_wait)
        end_time = time.time()
        duration = end_time - start_time
        total_size = args.size * 32.0 / 1000000000
        bandwidth = total_size / duration
        total_bw += bandwidth
        print('Bandwidth = {} Gb/s\tlatency = {} ms'.format(bandwidth, duration * 1000.0))
    print('Receiver is done')
    print('Avg bandwidth = {} Gb/s'.format(total_bw / args.niter))


def sender():
    print('This is sender with rank = {}'.format(args.rank))
    device = torch.device('cpu') if args.cpu else torch.device('cuda:0')
    tensor = torch.ones(args.size, device=device)
    # send a tensor to rank 1
    handle = bps.send_async(tensor, 1, name='test')
    bps.synchronize(handle)
    total_bw = 0
    for i in range(args.niter):
        start_time = time.time()
        handle = bps.send_async(tensor, 1, name='test')
        bps.synchronize(handle, args.busy_wait)
        end_time = time.time()
        duration = end_time - start_time
        total_size = args.size * 32.0 / 1000000000
        bandwidth = total_size / duration
        total_bw += bandwidth
        print('Bandwidth = {} Gb/s\tlatency = {} ms'.format(bandwidth, duration * 1000.0))
    print('Sender is done')
    print('Avg bandwidth = {} Gb/s'.format(total_bw / args.niter))

def get_nodes():
    # below is an example of $ARNOLD_WORKER_HOSTS:
    # 192.168.1.1:9000,192.168.1.2:9000,192.168.1.2:9001, 192.168.1.3:9003	
    host_ports = os.environ.get('ARNOLD_WORKER_HOSTS', None)
    host_ports = host_ports.split(',')
    if args.rank == 0:
        print('All hosts:', host_ports)
    return host_ports

def main():
    get_arguments()
    host_ports = get_nodes()
    bps.init(lazy=False, worker_id=args.rank)
    if args.rank == 0:
        sender()
    elif args.rank == 1:
        receiver()
    else:
        raise ValueError("unexpected rank")
    bps.shutdown()

if __name__ == "__main__":
    main()