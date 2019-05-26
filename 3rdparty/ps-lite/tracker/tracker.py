"""
Tracker script for DMLC
Implements the tracker control protocol
 - start dmlc jobs
 - start ps scheduler and rabit tracker
 - help nodes to establish links with each other

Tianqi Chen
"""

import sys
import os
import socket
import struct
import subprocess
import time
import logging
import random
from threading import Thread

"""
Extension of socket to handle recv and send of special data
"""
class ExSocket:
    def __init__(self, sock):
        self.sock = sock
    def recvall(self, nbytes):
        res = []
        sock = self.sock
        nread = 0
        while nread < nbytes:
            chunk = self.sock.recv(min(nbytes - nread, 1024))
            nread += len(chunk)
            res.append(chunk)
        return ''.join(res)
    def recvint(self):
        return struct.unpack('@i', self.recvall(4))[0]
    def sendint(self, n):
        self.sock.sendall(struct.pack('@i', n))
    def sendstr(self, s):
        self.sendint(len(s))
        self.sock.sendall(s)
    def recvstr(self):
        slen = self.recvint()
        return self.recvall(slen)

# magic number used to verify existence of data
kMagic = 0xff99

class SlaveEntry:
    def __init__(self, sock, s_addr):
        slave = ExSocket(sock)
        self.sock = slave
        self.host = socket.gethostbyname(s_addr[0])
        magic = slave.recvint()
        assert magic == kMagic, 'invalid magic number=%d from %s' % (magic, self.host)
        slave.sendint(kMagic)
        self.rank = slave.recvint()
        self.world_size = slave.recvint()
        self.jobid = slave.recvstr()
        self.cmd = slave.recvstr()

    def decide_rank(self, job_map):
        if self.rank >= 0:
            return self.rank
        if self.jobid != 'NULL' and self.jobid in job_map:
            return job_map[self.jobid]
        return -1

    def assign_rank(self, rank, wait_conn, tree_map, parent_map, ring_map):
        self.rank = rank
        nnset = set(tree_map[rank])
        rprev, rnext = ring_map[rank]
        self.sock.sendint(rank)
        # send parent rank
        self.sock.sendint(parent_map[rank])
        # send world size
        self.sock.sendint(len(tree_map))
        self.sock.sendint(len(nnset))
        # send the rprev and next link
        for r in nnset:
            self.sock.sendint(r)
        # send prev link
        if rprev != -1 and rprev != rank:
            nnset.add(rprev)
            self.sock.sendint(rprev)
        else:
            self.sock.sendint(-1)
        # send next link
        if rnext != -1 and rnext != rank:
            nnset.add(rnext)
            self.sock.sendint(rnext)
        else:
            self.sock.sendint(-1)
        while True:
            ngood = self.sock.recvint()
            goodset = set([])
            for i in xrange(ngood):
                goodset.add(self.sock.recvint())
            assert goodset.issubset(nnset)
            badset = nnset - goodset
            conset = []
            for r in badset:
                if r in wait_conn:
                    conset.append(r)
            self.sock.sendint(len(conset))
            self.sock.sendint(len(badset) - len(conset))
            for r in conset:
                self.sock.sendstr(wait_conn[r].host)
                self.sock.sendint(wait_conn[r].port)
                self.sock.sendint(r)
            nerr = self.sock.recvint()
            if nerr != 0:
                continue
            self.port = self.sock.recvint()
            rmset = []
            # all connection was successuly setup
            for r in conset:
                wait_conn[r].wait_accept -= 1
                if wait_conn[r].wait_accept == 0:
                    rmset.append(r)
            for r in rmset:
                wait_conn.pop(r, None)
            self.wait_accept = len(badset) - len(conset)
            return rmset

class RabitTracker:
    """
    tracker for rabit
    """
    def __init__(self, hostIP, nslave, port = 9091, port_end = 9999):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind((hostIP, port))
                self.port = port
                break
            except socket.error:
                continue
        sock.listen(16)
        self.sock = sock
        self.hostIP = hostIP
        logging.info('start listen on %s:%d' % (hostIP, self.port))

    def __del__(self):
        self.sock.close()

    def slave_envs(self):
        """
        get enviroment variables for slaves
        can be passed in as args or envs
        """
        return {'DMLC_TRACKER_URI': self.hostIP,
                'DMLC_TRACKER_PORT': self.port}

    def get_neighbor(self, rank, nslave):
        rank = rank + 1
        ret = []
        if rank > 1:
            ret.append(rank / 2 - 1)
        if rank * 2 - 1  < nslave:
            ret.append(rank * 2 - 1)
        if rank * 2 < nslave:
            ret.append(rank * 2)
        return ret

    def get_tree(self, nslave):
        tree_map = {}
        parent_map = {}
        for r in range(nslave):
            tree_map[r] = self.get_neighbor(r, nslave)
            parent_map[r] = (r + 1) / 2 - 1
        return tree_map, parent_map

    def find_share_ring(self, tree_map, parent_map, r):
        """
        get a ring structure that tends to share nodes with the tree
        return a list starting from r
        """
        nset = set(tree_map[r])
        cset = nset - set([parent_map[r]])
        if len(cset) == 0:
            return [r]
        rlst = [r]
        cnt = 0
        for v in cset:
            vlst = self.find_share_ring(tree_map, parent_map, v)
            cnt += 1
            if cnt == len(cset):
                vlst.reverse()
            rlst += vlst
        return rlst

    def get_ring(self, tree_map, parent_map):
        """
        get a ring connection used to recover local data
        """
        assert parent_map[0] == -1
        rlst = self.find_share_ring(tree_map, parent_map, 0)
        assert len(rlst) == len(tree_map)
        ring_map = {}
        nslave = len(tree_map)
        for r in range(nslave):
            rprev = (r + nslave - 1) % nslave
            rnext = (r + 1) % nslave
            ring_map[rlst[r]] = (rlst[rprev], rlst[rnext])
        return ring_map

    def get_link_map(self, nslave):
        """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
        tree_map, parent_map = self.get_tree(nslave)
        ring_map = self.get_ring(tree_map, parent_map)
        rmap = {0 : 0}
        k = 0
        for i in range(nslave - 1):
            k = ring_map[k][1]
            rmap[k] = i + 1

        ring_map_ = {}
        tree_map_ = {}
        parent_map_ ={}
        for k, v in ring_map.items():
            ring_map_[rmap[k]] = (rmap[v[0]], rmap[v[1]])
        for k, v in tree_map.items():
            tree_map_[rmap[k]] = [rmap[x] for x in v]
        for k, v in parent_map.items():
            if k != 0:
                parent_map_[rmap[k]] = rmap[v]
            else:
                parent_map_[rmap[k]] = -1
        return tree_map_, parent_map_, ring_map_

    def handle_print(self,slave, msg):
        logging.info(msg.strip())
        sys.stdout.write(msg)
        sys.stdout.flush()

    def accept_slaves(self, nslave):
        # set of nodes that finishs the job
        shutdown = {}
        # set of nodes that is waiting for connections
        wait_conn = {}
        # maps job id to rank
        job_map = {}
        # list of workers that is pending to be assigned rank
        pending = []
        # lazy initialize tree_map
        tree_map = None

        while len(shutdown) != nslave:
            fd, s_addr = self.sock.accept()
            s = SlaveEntry(fd, s_addr)
            if s.cmd == 'print':
                msg = s.sock.recvstr()
                self.handle_print(s, msg)
                continue
            if s.cmd == 'shutdown':
                assert s.rank >= 0 and s.rank not in shutdown
                assert s.rank not in wait_conn
                shutdown[s.rank] = s
                logging.debug('Recieve %s signal from %d' % (s.cmd, s.rank))
                continue
            assert s.cmd == 'start' or s.cmd == 'recover'
            # lazily initialize the slaves
            if tree_map == None:
                assert s.cmd == 'start'
                if s.world_size > 0:
                    nslave = s.world_size
                tree_map, parent_map, ring_map = self.get_link_map(nslave)
                # set of nodes that is pending for getting up
                todo_nodes = range(nslave)
            else:
                assert s.world_size == -1 or s.world_size == nslave
            if s.cmd == 'recover':
                assert s.rank >= 0

            rank = s.decide_rank(job_map)
            # batch assignment of ranks
            if rank == -1:
                assert len(todo_nodes) != 0
                pending.append(s)
                if len(pending) == len(todo_nodes):
                    pending.sort(key = lambda x : x.host)
                    for s in pending:
                        rank = todo_nodes.pop(0)
                        if s.jobid != 'NULL':
                            job_map[s.jobid] = rank
                        s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                        if s.wait_accept > 0:
                            wait_conn[rank] = s
                        logging.debug('Recieve %s signal from %s; assign rank %d' % (s.cmd, s.host, s.rank))
                if len(todo_nodes) == 0:
                    logging.info('@tracker All of %d nodes getting started' % nslave)
                    self.start_time = time.time()
            else:
                s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                logging.debug('Recieve %s signal from %d' % (s.cmd, s.rank))
                if s.wait_accept > 0:
                    wait_conn[rank] = s
        logging.info('@tracker All nodes finishes job')
        self.end_time = time.time()
        logging.info('@tracker %s secs between node start and job finish' % str(self.end_time - self.start_time))

    def start(self, nslave):
        def run():
            self.accept_slaves(nslave)
        self.thread = Thread(target = run, args = ())
        self.thread.setDaemon(True)
        self.thread.start()

    def join(self):
        while self.thread.isAlive():
            self.thread.join(100)

class PSTracker:
    """Start the schduler node in PS
    """
    def __init__(self, hostIP, cmd, port = 9091, port_end = 9999, envs = {}):
        """
        Starts the PS scheduler
        """
        self.cmd = cmd
        if cmd is None:
            return

        self.hostIP = hostIP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        max_retry = 100

        for i in range(0, max_retry):
            if i + 1 == max_retry:
                raise Exception('the schduler is faild to bind a port')
            port = random.randint(port, port_end)
            try:
                sock.bind(('', port))
                self.port = port
                sock.close()
                break
            except socket.error:
                continue
        env = os.environ.copy()

        env['DMLC_ROLE'] = 'scheduler'
        env['DMLC_PS_ROOT_URI'] = str(self.hostIP)
        env['DMLC_PS_ROOT_PORT'] = str(self.port)
        for k, v in envs.items():
            env[k] = str(v)
        self.thread = Thread(target = (lambda : subprocess.check_call(self.cmd, env=env, shell=True)), args = ())
        self.thread.setDaemon(True)
        self.thread.start()

    def join(self):
        if self.cmd is not None:
            while self.thread.isAlive():
                self.thread.join(100)

    def slave_envs(self):
        if self.cmd is None:
            return {}
        else:
            return {'DMLC_PS_ROOT_URI': self.hostIP,
                    'DMLC_PS_ROOT_PORT': self.port}

def submit(nworker, nserver, fun_submit, hostIP = 'auto', pscmd = None):
    """submit job

    Paramaters
    ----------
    nworker : int
        number of workers
    nserver : int
        number of servers, if 0 then submit as rabit a job, otherwise sumbit as
        a parmaeter server job
    fun_sumbit : func
        the function to submit the jobs for servers and workers
    hostIP : str, optional
        the host ip of the root node
    pscmd :
    """
    # get the root node ip
    if hostIP == 'auto':
        hostIP = 'ip'
    if hostIP == 'dns':
        hostIP = socket.getfqdn()
    elif hostIP == 'ip':
        from socket import gaierror
        try:
            hostIP = socket.gethostbyname(socket.getfqdn())
        except gaierror:
            logging.warn('gethostbyname(socket.getfqdn()) failed... trying on hostname()')
            hostIP = socket.gethostbyname(socket.gethostname())
        if hostIP.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # doesn't have to be reachable
            s.connect(('10.255.255.255', 0))
            hostIP = s.getsockname()[0]

    envs = {'DMLC_NUM_WORKER' : nworker,
            'DMLC_NUM_SERVER' : nserver}

    # start the root
    if nserver == 0:
        rabit = RabitTracker(hostIP = hostIP, nslave = nworker)
        envs.update(rabit.slave_envs())
        rabit.start(nworker)
    else:
        pserver = PSTracker(hostIP = hostIP, cmd=pscmd, envs = envs)
        envs.update(pserver.slave_envs())

    # start the workers and servers
    fun_submit(nworker, nserver, envs)

    # wait the root finished
    if nserver == 0:
        rabit.join()
    else:
        pserver.join()

def config_logger(args):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    level = args.log_level if 'log_level' in args else 'DEBUG'
    level = eval('logging.' + level)
    if 'log_file' not in args or args.log_file is None:
        logging.basicConfig(format=FORMAT, level = level)
    else:
        logging.basicConfig(format=FORMAT, level = level, filename = args.log_file)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(FORMAT))
        console.setLevel(level)
        logging.getLogger('').addHandler(console)
