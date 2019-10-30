#!/usr/bin/python

from __future__ import print_function
import os
import subprocess
import threading
import sys
import time
import traceback

def worker(local_rank, local_size, command):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)

    if os.getenv("BYTEPS_ENABLE_GDB", 0):
        if command.find("python") != 0:
            command = "python " + command
        command = "gdb -ex 'run' -ex 'bt' -batch --args " + command

    ## only enable profiling for the worker with rank of 0
    # if os.environ.get("DMLC_WORKER_ID") == "0" and local_rank == 0:
    #     # my_env["BYTEPS_LOG_LEVEL"] = "TRACE"
    #     my_env["TRACE_ON"] = "ON"
    #     my_env["TRACE_END_STEP"] = "110"
    #     my_env["TRACE_DIR"]= "./traces"
    #     print("\n!!!Enable profiling for WORKER_ID: 0 and local_rank: 0!!!\nCommand: %s\n" % command)
    #     sys.stdout.flush()
    if os.environ.get("TRACE_ON", "") == "1":
        print("\n!!!Enable profiling for WORKER_ID: %s and local_rank: %d!!!" % (os.environ.get("DMLC_WORKER_ID"), local_rank))
        print("TRACE_END_STEP: %s\t TRACE_DIR: %s" % (os.environ.get("TRACE_END_STEP", ""), os.environ.get("TRACE_DIR", "")))
        print("Command: %s\n" % command)
        sys.stdout.flush()
    # command = "nvprof -f -o /mnt/cephfs_new_wj/mlsys/huhanpeng/prof/worker_" + str(os.environ["DMLC_WORKER_ID"]) + "_"  + str(local_rank) + ".nvvp " + command
    subprocess.check_call(command, env=my_env, stdout=sys.stdout, stderr=sys.stderr, shell=True)

if __name__ == "__main__":
    DMLC_ROLE = os.environ.get("DMLC_ROLE")
    print("BytePS launching " + (DMLC_ROLE if DMLC_ROLE else 'None'))
    BYTEPS_SERVER_MXNET_PATH = os.getenv("BYTEPS_SERVER_MXNET_PATH")
    print("BYTEPS_SERVER_MXNET_PATH: " + (BYTEPS_SERVER_MXNET_PATH if BYTEPS_SERVER_MXNET_PATH else 'None'))
    sys.stdout.flush()

    if os.environ["DMLC_ROLE"] == "worker":
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            local_size = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            local_size = 1
        t = [None] * local_size
        for i in range(local_size):
            command = ' '.join(sys.argv[1:])
            t[i] = threading.Thread(target=worker, args=[i, local_size, command])
            t[i].daemon = True
            t[i].start()

        for i in range(local_size):
            t[i].join()

    else:
        if "BYTEPS_SERVER_MXNET_PATH" not in os.environ:
            print("BYTEPS_SERVER_MXNET_PATH env not set")
            sys.stdout.flush()
            os._exit(0)
        sys.path.insert(0, os.getenv("BYTEPS_SERVER_MXNET_PATH")+"/python")
        import mxnet