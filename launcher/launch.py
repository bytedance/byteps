#!/usr/bin/python3

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
# TODO(yulu): temp solution. Using DMLC_WORKER_ID as the node id for now. Will
# use PHY_NODE_ID later.
# WORKER_REQUIRED_ENVS = ["PHY_NODE_ID"]
WORKER_REQUIRED_ENVS = ["DMLC_WORKER_ID"]
NUMA_PATH = "/sys/devices/system/node"


def get_numa_info():
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
                ret.append(cpu_ids)
    else:
        print("NUMA PATH %s NOT FOUND" % NUMA_PATH)
    return ret


def allocate_cpu(local_size):
    def _get_allocation(nodes, quota):
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
            ret.append(node[last_idx:last_idx+quota])
            for idx in sorted(range(quota_bck), reverse=True):
                del node[idx]
            return ret
        return ret

    def _get_quota(nodes, local_size):
        if len(nodes) > 1:
            cpu_nums = reduce(lambda x, y: (len(x) + len(y)), nodes)
        else:
            cpu_nums = len(nodes[0])
        
        # default quota is the number of cpus for non-root processess
        default_quota = int(os.getenv("BYTEPS_NUMA_DEFAULT_QUOTA", 6))
        while default_quota >= 1 and default_quota * local_size > cpu_nums:
            default_quota -= 2

        # root quota is the number of cpus for root processess
        # root does more work, thus using more cpus
        root_quota = cpu_nums - default_quota * (local_size - 1)
        if int(os.getenv("BYTEPS_NUMA_ROOT_QUOTA", 0)):
            root_quota = int(os.getenv("BYTEPS_NUMA_ROOT_QUOTA", 0))

        node_size = len(nodes[0])
        while root_quota >= 1 and root_quota > node_size:
            root_quota -= 2
        return [default_quota] * (local_size - 1) + [root_quota]

    nodes = get_numa_info()
    if not nodes:
        return None
    quota_list = _get_quota(nodes, local_size)
    ret = []
    for quota in quota_list:
        while quota > 0:
            allocation = _get_allocation(nodes, quota)
            if allocation:
                ret.append(allocation)
                break
            else:
                quota -= 2

    return ret


def check_env():
    assert "DMLC_ROLE" in os.environ and \
           os.environ["DMLC_ROLE"].lower() in ["worker", "server", "scheduler", "joint"]
    required_envs = COMMON_REQUIRED_ENVS
    if os.environ["DMLC_ROLE"] in ["worker", "joint"]:
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


def worker_fn(local_rank, local_size, command, allocation=None):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)

    # TODO(yulu): temp solution. We should really set DMLC_NUM_WORKER and
    # DMLC_NUM_SERVER properly before running this script.
    my_env["DMLC_NUM_WORKER"] = str(int(os.environ["DMLC_NUM_WORKER"]) * local_size)
    my_env["DMLC_NUM_SERVER"] = my_env["DMLC_NUM_WORKER"]

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
    # node_id = int(os.environ.get("PHY_NODE_ID", "0"))
    # for running on arnold
    node_id = int(os.environ.get("DMLC_WORKER_ID", "0"))
    global_rank = int(node_id * local_size + local_rank)
    my_env["DMLC_WORKER_ID"] = str(global_rank)
    my_env["DMLC_RANK"] = my_env["DMLC_WORKER_ID"]
    log_file_name =os.getenv('BYTEPS_LOG_FILE', '')
    if log_file_name:
        my_file = open(f"{log_file_name}-g{global_rank}-l{local_rank}.log", "w+")
        stdout_sink = stderr_sink = my_file
    else:
        stdout_sink = sys.stdout
        stderr_sink = sys.stderr

    subprocess.check_call(command, env=my_env,
                          stdout=stdout_sink, stderr=stderr_sink, shell=True)

def server_fn(local_rank, local_size, command, allocation=None):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)
    if int(os.getenv("BYTEPS_ENABLE_GDB", 0)):
        command = "gdb -ex 'run' -ex 'bt' -batch --args " + command
    node_id = int(os.environ.get("BYTEPS_PHY_NODE_ID", "0"))
    global_rank = node_id * local_size + local_rank
    my_env["DMLC_RANK"] = str(node_id)
    subprocess.check_call(command, env=my_env,
                          stdout=sys.stdout, stderr=sys.stderr, shell=True)

def launch_bps():
    print("BytePS launching " + os.environ["DMLC_ROLE"])
    sys.stdout.flush()
    check_env()
    os.environ["PYTHONUNBUFFERED"] = "1"
    if os.environ["DMLC_ROLE"] in ["worker", "joint"]:
        if os.getenv("BYTEPS_FORCE_JOINT_MODE", "0").lower() in ["1", "true"]:
            os.environ["DMLC_ROLE"] = "joint"
        # launch workers
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            local_size = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            local_size = 1
        t = [None] * local_size

        if os.environ.get("BYTEPS_NUMA_ON", "") == "1":
            allocations = allocate_cpu(local_size)

        for i in range(local_size):
            command = ' '.join(sys.argv[1:])
            if os.environ.get("BYTEPS_NUMA_ON", "") == "1":
                t[i] = PropagatingThread(target=worker_fn, args=[
                    i, local_size, command, allocations[i]])
            else:
                t[i] = PropagatingThread(target=worker_fn, args=[
                    i, local_size, command])
            t[i].daemon = True
            t[i].start()

        for i in range(local_size):
            t[i].join()
        return

    if os.environ.get("BYTEPS_FORCE_DISTRIBUTED", "0") == "0" and \
       int(os.environ.get("DMLC_NUM_WORKER", "1")) == 1:
           # there's only one worker, and not forcing distributed mode
           return

    command = "python3 -c 'import byteps.server'"
    if os.environ["DMLC_ROLE"] == "scheduler":
        my_env = os.environ.copy()
        # TODO(yulu): temp solution. We should really set DMLC_NUM_WORKER and
        # DMLC_NUM_SERVER properly before running this script.
        my_env["PS_VERBOSE"] = os.environ.get("PS_VERBOSE", '1')
        my_env["DMLC_NUM_WORKER"] = str(int(os.environ["DMLC_NUM_WORKER"]) *
                                        int(os.environ["ARNOLD_WORKER_GPU"]))
        my_env["DMLC_NUM_SERVER"] = my_env["DMLC_NUM_WORKER"]
        subprocess.check_call(command, env=my_env,
                              stdout=sys.stdout, stderr=sys.stderr, shell=True)
        return

    if os.getenv("BYTEPS_FORCE_JOINT_MODE", "0").lower() in ["1", "true"]:
        # do nothing when DMLC_ROLE is "server".
        return

    # now it's the servers in non-colocate mode
    local_size = 1

    t = [None] * local_size
    for i in range(local_size):
        t[i] = PropagatingThread(target=server_fn, args=[
                i, local_size, command])
        t[i].daemon = True
        t[i].start()
    for i in range(local_size):
        t[i].join()

if __name__ == "__main__":
    launch_bps()
