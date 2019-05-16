#!/usr/bin/python

import os
import sys
import re
import time

whitelist_wildcards = [
    "DMLC_*",
    "PS_VERBOSE",
    "BYTEPS_*"
]

wildcards = [re.compile(x) for x in whitelist_wildcards]

if __name__ == "__main__":
    print "BytePS launching " + os.environ["DMLC_ROLE"]
    sys.stdout.flush()

    if os.environ["DMLC_ROLE"] == "worker":
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
        os.system("mpirun " + env_str + command_line)

    else:
        sys.path.insert(0, os.getenv("BYTEPS_SERVER_MXNET_PATH")+"/python")
        import mxnet
        print "BytePS Server MXNet version: " + mxnet.__version__
        # TODO: terminates when workers quit
        while True:
            time.sleep(3600)