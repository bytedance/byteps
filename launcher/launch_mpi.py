#!/usr/bin/python

import os
import sys
import re

whitelist_wildcards = [
    "DMLC_*"
]

wildcards = [re.compile(x) for x in whitelist_wildcards]

if __name__ == "__main__":
    env_str = ""

    for env in os.environ:
        whitelisted = False
        for w in wildcards:
            if re.match(w, env):
                whitelisted = True
                break
        if whitelisted:
            env_str += "-x %s " % env

    command_line = sys.argv[1:]

    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        gpu = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
    else:
        gpu = 1

    os.system(
        "mpirun --allow-run-as-root -n %d -mca pml ob1 -x NCCL_DEBUG=INFO " % gpu + 
        env_str + " " +
        " ".join(command_line))

