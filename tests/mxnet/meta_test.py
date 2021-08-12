# Copyright 2020 Amazon Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy
import time
import os
import subprocess
import sys
import threading

import byteps.mxnet as bps


class MetaTest(type):
    BASE_ENV = {"DMLC_NUM_WORKER": "1",
                "DMLC_NUM_SERVER": "1",
                "DMLC_PS_ROOT_URI": "127.0.0.1",
                "DMLC_PS_ROOT_PORT": "1234",
                "BYTEPS_LOG_LEVEL": "INFO",
                "BYTEPS_MIN_COMPRESS_BYTES": "0",
                "BYTEPS_PARTITION_BYTES": "2147483647"}
    for name, value in os.environ.items():
        if name not in BASE_ENV:
            BASE_ENV[name] = value
    SCHEDULER_ENV = copy.copy(BASE_ENV)
    SCHEDULER_ENV.update(DMLC_ROLE="scheduler")
    SERVER_ENV = copy.copy(BASE_ENV)
    SERVER_ENV.update(DMLC_ROLE="server")

    def __new__(cls, name, bases, dict):
        # decorate all test cases
        for k, v in dict.items():
            if k.startswith("test_") and hasattr(v, "__call__"):
                dict[k] = cls.launch_bps(v)

        for k, v in cls.BASE_ENV.items():
            os.environ[k] = v
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
        os.environ["DMLC_WORKER_ID"] = "0"
        os.environ["DMLC_ROLE"] = "worker"
        os.environ["BYTEPS_THREADPOOL_SIZE"] = "4"
        os.environ["BYTEPS_FORCE_DISTRIBUTED"] = "1"
        os.environ["BYTEPS_LOCAL_RANK"] = "0"
        os.environ["BYTEPS_LOCAL_SIZE"] = "1"
        return type(name, bases, dict)

    @classmethod
    def launch_bps(cls, func):
        def wrapper(*args, **kwargs):
            def run(env):
                subprocess.check_call(args=["bpslaunch"], shell=True,
                                      stdout=sys.stdout, stderr=sys.stderr,
                                      env=env)
                
            print("bps init")
            scheduler = threading.Thread(target=run,
                                         args=(cls.SCHEDULER_ENV,))
            server = threading.Thread(target=run, args=(cls.SERVER_ENV,))
            scheduler.daemon = True
            server.daemon = True
            scheduler.start()
            server.start()

            bps.init()
            func(*args, **kwargs)
            bps.shutdown()

            scheduler.join()
            server.join()
            print("bps shutdown")
            time.sleep(2)

        return wrapper
