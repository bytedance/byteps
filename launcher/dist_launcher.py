#!/usr/bin/python
"""
Launch a distributed job for BytePS
"""
import argparse
import os, sys
import signal
import logging
import subprocess
from multiprocessing import Pool, Process
from threading import Thread

def preprocess_envs(args_envs):
    envs_map = {}
    for item in args_envs:
        i = item.find(":")
        if i != -1:
            key = item[:i]
            val = item[i+1:]
        envs_map[key] = val
    return envs_map

def get_env(envs_map):
    envs = []
    # get system envs
    keys = ['OMP_NUM_THREADS', 'KMP_AFFINITY']
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            envs.append('export ' + k + '=' + v + ';')
    # get ass_envs
    for k, v in envs_map.items():
        envs.append('export ' + str(k) + '=' + str(v) + ';')
    return (' '.join(envs))

def get_hosts_from_file(filename):
    with open(filename) as f:
        tmp = f.readlines()
    assert len(tmp) > 0
    hosts = []
    for h in tmp:
        if len(h.strip()) > 0:
            # parse addresses of the form ip:port
            h = h.strip()
            i = h.find(":")
            p = "22"
            if i != -1:
                p = h[i+1:]
                h = h[:i]
            # hosts now contain the pair ip, port
            hosts.append((h, p))
    return hosts


def start_ssh(prog, node, port, username, fname):
    def run(prog):
        subprocess.check_call(prog, shell=True)

    dirname = 'sshlog'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    pname = dirname + '/' + fname
    if username is not None:
        prog = 'ssh -o StrictHostKeyChecking=no ' + ' -l ' + username \
               + ' ' + node + ' -p ' + port + ' \'' + prog + '\'' \
               + ' > ' + pname + '.stdout' + ' 2>' + pname + '.stderr&'
    else:
        prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + port + ' \'' + prog + '\'' \
               + ' > ' + pname + '.stdout' + ' 2>' + pname + '.stderr&'

    thread = Thread(target=run, args=(prog,))
    thread.setDaemon(True)
    thread.start()
    return thread


def submit(args):
    worker_hosts = get_hosts_from_file(args.worker_hostfile)
    server_hosts = get_hosts_from_file(args.server_hostfile)
    num_worker = len(worker_hosts)
    num_server = len(server_hosts)
    assert num_worker >= 1
    assert num_server >= 1
    print('Launch %d workers and %d servers' % (num_worker, num_server))

    # common env
    pass_envs = preprocess_envs(args.env)
    pass_envs['DMLC_NUM_WORKER'] = str(num_worker)
    pass_envs['DMLC_NUM_SERVER'] = str(num_server)
    pass_envs['DMLC_INTERFACE'] = str(args.interface)
    pass_envs['DMLC_PS_ROOT_URI'] = str(args.scheduler_ip)
    pass_envs['DMLC_PS_ROOT_PORT'] = str(args.scheduler_port)

    username = ''
    if args.username is not None:
        username = args.username

    threads = []
    for (node, port) in [(args.scheduler_ip, args.scheduler_ssh_port)]:
        name = 'scheduler'
        pass_envs['DMLC_ROLE'] = name
        prog = get_env(pass_envs) + (' '.join(args.command))
        threads.append(start_ssh(prog, node, port, username, name))
    for i, (node, port) in enumerate(worker_hosts):
        name = 'worker'
        pass_envs['DMLC_ROLE'] = name
        pass_envs['DMLC_WORKER_ID'] = str(i)
        prog = get_env(pass_envs) + (' '.join(args.command))
        threads.append(start_ssh(prog, node, port, username, name + str(i)))
    for i, (node, port) in enumerate(server_hosts):
        name = 'server'
        pass_envs['DMLC_ROLE'] = name
        prog = get_env(pass_envs) + (' '.join(args.command))
        threads.append(start_ssh(prog, node, port, username, name + str(i)))

    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser(description='Launch a distributed training job for BytePS')
    parser.add_argument('-WH', '--worker-hostfile', required=True, type=str,
                        help = 'the hostfile of worker machines which will run the job.')
    parser.add_argument('-SH', '--server-hostfile', required=True, type=str,
                        help = 'the hostfile of server machines which will run the job.')
    parser.add_argument('--scheduler-ip', required=True, type=str,
                        help = 'the ip address of the scheduler')
    parser.add_argument('--scheduler-port', required=True, type=int,
                        help = 'the port of the scheduler')
    parser.add_argument('--interface', type=str, default='eth0',
                        help = 'the network interface to use')
    parser.add_argument('--env', action='append', default=[],
                        help = 'Given a pair of environment_variable:value, sets this value of \
                        environment variable for all workers and servers. Example OMP_NUM_THREADS:3')
    parser.add_argument('--username', type=str,
                        help = 'the username for ssh')
    parser.add_argument('--scheduler-ssh-port', type=str, default='22',
                        help = 'the ssh port of the scheduler')
    parser.add_argument('command', nargs='+',
                        help = 'command for launching the program')

    args = parser.parse_args()

    # check necessary args
    assert args.worker_hostfile
    assert args.server_hostfile
    assert args.scheduler_ip
    assert args.scheduler_port

    submit(args)


def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()
