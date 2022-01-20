#!/usr/bin/python

from __future__ import print_function
import os
import re
import subprocess
import threading
import sys
import time
from functools import reduce


class PropagatingThread(threading.Thread):
    """ propagate exceptions to the parent's thread
    refer to https://stackoverflow.com/a/31614591/9601110
    """

    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                #  python 2.x
                self.ret = self._Thread__target(
                    *self._Thread__args, **self._Thread__kwargs)
            else:
                # python 3.x
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.exc


COMMON_REQUIRED_ENVS = ["DMLC_ROLE", "DMLC_NUM_WORKER", "DMLC_NUM_SERVER",
                        "DMLC_PS_ROOT_URI", "DMLC_PS_ROOT_PORT"]
WORKER_REQUIRED_ENVS = ["DMLC_WORKER_ID"]
NUMA_PATH = "/sys/devices/system/node"

def allocate_cpu(local_size):
    cpu_mt = os.getenv("BYTEPS_MULTITHREADED_CPU", "1").lower() in ["1", "true"]
    def get_numa_info():
        """
        returns a list of list, each sub list is the cpu ids of a numa node. e.g
        [[0,1,2,3], [4,5,6,7]]
        """
        ret = []
        if os.path.exists(NUMA_PATH):
            items = os.listdir(NUMA_PATH)
            nodes = list(filter(lambda str: str.startswith("node"), items))
            if nodes:
                for node in nodes:
                    items = os.listdir(os.path.join(NUMA_PATH, node))
                    cpus = [re.findall("cpu\d+", cpu) for cpu in items]
                    cpus = list(filter(lambda x: x, cpus))
                    cpu_ids = [int(cpu[0].split('cpu')[1]) for cpu in cpus]
                    cpu_ids = sorted(cpu_ids)
                    if cpu_mt:
                        cpu_ids = cpu_ids[:len(cpu_ids) // 2]
                    ret.append(cpu_ids)
        else:
            print("NUMA PATH %s NOT FOUND" % NUMA_PATH)
        return ret

    def _get_allocation(nodes, quota, cpu_num, cpu_blacklist):
        if quota < 1:
            raise ValueError("quota should be no less than 1")
        ret = []
        for node in nodes:
            if len(node) < quota:
                continue
            split_index = []
            for i in range(1, quota):
                if node[i] != node[i-1] + 1:
                    split_index.append(i)
            quota_bck = quota
            last_idx = 0
            for idx in split_index:
                ret.append(node[last_idx:idx])
                quota -= idx - last_idx
                last_idx = idx
            curr_alloc = node[last_idx:last_idx+quota]
            curr_alloc = [item for item in curr_alloc if item not in cpu_blacklist]
            ret.append(curr_alloc)
            if cpu_mt:
                curr_alloc = [x + cpu_num for x in curr_alloc]
                curr_alloc = [item for item in curr_alloc if item not in cpu_blacklist]
                ret.append(curr_alloc)
            for idx in sorted(range(quota_bck), reverse=True):
                del node[idx]
            return ret
        return ret

    def _get_quota(nodes, local_size):

        # default quota is the number of physical cores for non-root processess
        default_quota = cpu_num // local_size
        default_quota = int(os.getenv("BYTEPS_NUMA_DEFAULT_QUOTA", default_quota))
        while default_quota >= 1 and default_quota * local_size > cpu_num:
            default_quota -= 1

        # root quota is the number of cpus for root processess
        # root does more work, thus using more cpus
        root_quota = cpu_num - default_quota * (local_size - 1)
        if int(os.getenv("BYTEPS_NUMA_ROOT_QUOTA", 0)):
            root_quota = int(os.getenv("BYTEPS_NUMA_ROOT_QUOTA", 0))

        node_size = len(nodes[0])
        if cpu_mt:
            node_size //= 2
        while root_quota >= 1 and root_quota > node_size:
            root_quota -= 1
        return [default_quota] * (local_size - 1) + [root_quota]

    nodes = get_numa_info()
    if not nodes:
        return None
    cpu_num = reduce(lambda x, y: (x + len(y)), nodes, 0)
    quota_list = _get_quota(nodes, local_size)
    cpu_blacklist = os.getenv("BYTEPS_CPU_BLACKLIST", "-1")
    cpu_blacklist = [int(item) for item in cpu_blacklist.split(",")]
    ret = []
    for quota in quota_list:
        while quota > 0:
            allocation = _get_allocation(nodes, quota, cpu_num, cpu_blacklist)
            if allocation:
                ret.append(allocation)
                break
            else:
                quota -= 1

    return ret


def check_env():
    assert "DMLC_ROLE" in os.environ and \
           os.environ["DMLC_ROLE"].lower() in ["worker", "server", "scheduler"]
    required_envs = COMMON_REQUIRED_ENVS
    if os.environ["DMLC_ROLE"] == "worker":
        assert "DMLC_NUM_WORKER" in os.environ
        num_worker = int(os.environ["DMLC_NUM_WORKER"])
        assert num_worker >= 1
        if num_worker == 1:
            required_envs = []
        required_envs += WORKER_REQUIRED_ENVS
    for env in required_envs:
        if env not in os.environ:
            print("The env " + env + " is missing")
            os._exit(0)


def worker(local_rank, local_size, command, allocation=None):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)
    if int(os.getenv("BYTEPS_ENABLE_GDB", 0)):
        if command.find("python") != 0:
            command = "python " + command
        command = "gdb -ex 'run' -ex 'bt' -batch --args " + command

    if allocation:
        print("enable NUMA finetune...")
        retval = subprocess.call(
            ["dpkg", "-s", "numactl"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if retval == 0:
            numa = "numactl --physcpubind "
            for cpu_set in allocation:
                if len(cpu_set) == 1:
                    numa += "{},".format(cpu_set[0])
                else:
                    numa += "{}-{},".format(cpu_set[0], cpu_set[-1])
            numa = numa.strip(',') + ' '
            command = numa + command
            print("Command: %s\n" % command)
        else:
            print("Warning: numactl not found. try `sudo apt-get install numactl`.")

    if os.environ.get("BYTEPS_TRACE_ON", "") == "1":
        print("\n!!!Enable profiling for WORKER_ID: %s and local_rank: %d!!!" %
              (os.environ.get("DMLC_WORKER_ID"), local_rank))
        print("BYTEPS_TRACE_START_STEP: %s\tBYTEPS_TRACE_END_STEP: %s\t BYTEPS_TRACE_DIR: %s" % (os.environ.get(
            "BYTEPS_TRACE_START_STEP", ""), os.environ.get("BYTEPS_TRACE_END_STEP", ""), os.environ.get("BYTEPS_TRACE_DIR", "")))
        print("Command: %s\n" % command)
        sys.stdout.flush()
        trace_path = os.path.join(os.environ.get(
            "BYTEPS_TRACE_DIR", "."), str(local_rank))
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)
    subprocess.check_call(command, env=my_env,
                          stdout=sys.stdout, stderr=sys.stderr, shell=True)

def parse_num_range(core_list):
    # core_list is a colon-seperated string. each section is the physical
    # core assignment for the corresponding byteps worker.
    # example input: 1,4-5,7-11,12:20-25
    # example output: [[[1], [4, 5], [7, 8, 9, 10, 11], [12]], [[20, 21, 22, 23, 24, 25]]]
    core_list = core_list.split(':')
    ret = []
    for item in core_list:
        temp = [(lambda sub: range(sub[0], sub[-1] + 1))(list(map(int, elem.split('-')))) for elem in item.split(',')]
        ret.append([list(a) for a in temp])
    return ret

def launch_bps():
    print("BytePS launching " + os.environ["DMLC_ROLE"])
    sys.stdout.flush()
    check_env()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["UCX_HANDLE_ERRORS"] = os.getenv("UCX_HANDLE_ERRORS", "none")
    if os.environ["DMLC_ROLE"] == "worker":
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            local_size = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            local_size = 1
        t = [None] * local_size

        bind_to_cores = os.getenv("BYTEPS_NUMA_ON", "1") == "1"
        if bind_to_cores:
            user_override = os.getenv("BYTEPS_VISIBLE_CPU_CORES", "").strip()
            if user_override:
                allocations = parse_num_range(user_override)
            else:
                allocations = allocate_cpu(local_size)

        for i in range(local_size):
            command = ' '.join(sys.argv[1:])
            if bind_to_cores:
                t[i] = PropagatingThread(target=worker, args=[
                    i, local_size, command, allocations[i]])
            else:
                t[i] = PropagatingThread(target=worker, args=[
                    i, local_size, command])
            t[i].daemon = True
            t[i].start()

        for i in range(local_size):
            t[i].join()

    elif os.environ.get("BYTEPS_FORCE_DISTRIBUTED", "") == "1" or \
         int(os.environ.get("DMLC_NUM_WORKER", "1")) > 1:
        command = "python3 -c 'import byteps.server'"
        if int(os.getenv("BYTEPS_ENABLE_GDB", 0)):
            command = "gdb -ex 'run' -ex 'bt' -batch --args " + command
        print("Command: %s\n" % command, flush=True)
        my_env = os.environ.copy()
        subprocess.check_call(command, env=my_env,
                              stdout=sys.stdout, stderr=sys.stderr, shell=True)


if __name__ == "__main__":
    launch_bps()
