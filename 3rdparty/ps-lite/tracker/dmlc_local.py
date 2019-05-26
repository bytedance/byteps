#!/usr/bin/env python
"""
DMLC submission script, local machine version
"""

import argparse
import sys
import os
import subprocess
from threading import Thread
import tracker
import signal
import logging

keepalive = """
nrep=0
rc=254
while [ $rc -eq 254 ];
do
    export DMLC_NUM_ATTEMPT=$nrep
    %s
    rc=$?;
    nrep=$((nrep+1));
done
"""

class LocalLauncher(object):
    def __init__(self, args, unknown):
        self.args = args
        self.cmd = ' '.join(args.command) + ' ' + ' '.join(unknown)

    def exec_cmd(self, cmd, role, pass_env):
        env = os.environ.copy()
        for k, v in pass_env.items():
            env[k] = str(v)

        env['DMLC_ROLE'] = role

        ntrial = 0
        while True:
            if os.name == 'nt':
                env['DMLC_NUM_ATTEMPT'] = str(ntrial)
                ret = subprocess.call(cmd, shell=True, env = env)
                if ret == 254:
                    ntrial += 1
                    continue
            else:
                bash = keepalive % (cmd)
                ret = subprocess.call(bash, shell=True, executable='bash', env = env)
            if ret == 0:
                logging.debug('Thread %d exit with 0')
                return
            else:
                if os.name == 'nt':
                    os.exit(-1)
                else:
                    raise Exception('Get nonzero return code=%d' % ret)

    def submit(self):
        def mthread_submit(nworker, nserver, envs):
            """
            customized submit script
            """
            procs = {}
            for i in range(nworker + nserver):
                role = 'worker' if i < nworker else 'server'
                procs[i] = Thread(target = self.exec_cmd, args = (self.cmd, role, envs))
                procs[i].setDaemon(True)
                procs[i].start()
        return mthread_submit

    def run(self):
        tracker.config_logger(self.args)
        tracker.submit(self.args.num_workers,
                       self.args.num_servers,
                       fun_submit = self.submit(),
                       pscmd = self.cmd)

def main():
    parser = argparse.ArgumentParser(
        description='DMLC script to submit dmlc jobs as local process')

    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help = 'number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help = 'number of server nodes to be launched')
    parser.add_argument('--log-level', default='INFO', type=str,
                        choices=['INFO', 'DEBUG'],
                        help = 'logging level')
    parser.add_argument('--log-file', type=str,
                        help = 'output log to the specific log file')
    parser.add_argument('command', nargs='+',
                        help = 'command for launching the program')
    args, unknown = parser.parse_known_args()

    launcher = LocalLauncher(args, unknown)
    launcher.run()

if __name__ == '__main__':
    main()
