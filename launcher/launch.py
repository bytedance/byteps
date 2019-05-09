#!/usr/bin/python

import os
import subprocess
import threading
import sys

def worker(local_rank, local_size, command):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)
    subprocess.check_call(command, env=my_env, shell=True)

if __name__ == "__main__":
    print "BytePS launching " + os.environ["DMLC_ROLE"]
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
        os.system("python /opt/tiger/byteps/example/mxnet/train_imagenet_horovod.py")

