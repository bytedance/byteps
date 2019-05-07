#!/usr/bin/python

import os
import sys

if __name__ == "__main__":
    print "BytePS launching " + os.environ["DMLC_ROLE"]

    if os.environ["DMLC_ROLE"] == "worker":
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            gpu = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            gpu = 1
        for i in range(gpu):
            os.system(
                "BYTEPS_LOCAL_RANK=%d BYTEPS_LOCAL_SIZE=%d %s" % (i, gpu, ' '.join(sys.argv[1:])))

    else:
        os.system("python /opt/tiger/byteps/example/mxnet/train_imagenet_horovod.py")
