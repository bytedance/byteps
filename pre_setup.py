# For internal use. Please do not modify this file.
import os
import sys
import subprocess
import pathlib
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError, DistutilsSetupError

def execute_cmd(cmd):
    make_process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
    make_process.communicate()
    if make_process.returncode:
        raise DistutilsSetupError('An ERROR occured while running the command: \n' +
            cmd + '\n Exit code: {0}'.format(make_process.returncode))

def setup():
    # install NCCL
    execute_cmd("http_proxy= apt update " +
              "&& apt install nccl-repo-ubuntu1804-2.4.8-ga-cuda10.1 " +
              "&& apt-key add /var/nccl-repo-2.4.8-ga-cuda10.1/7fa2af80.pub " +
              "&& apt update && apt install libnccl2=2.4.8-1+cuda10.1 libnccl-dev=2.4.8-1+cuda10.1")

    # add apt dependency
    apt_install_command = "apt-get update && apt-get install -y "
    apt_deps = ['librdmacm-dev', 
                'libnuma-dev', 
                'libjemalloc-dev']
    for item in apt_deps:
        apt_install_command += item + " "
    execute_cmd(apt_install_command)
    os.environ['BYTEPS_CUDA_HOME'] = '/opt/tiger/cuda'
    os.environ['BYTEPS_NCCL_HOME'] = '/opt/tiger/cuda'
    return

def extra_make_option():
    if os.path.isfile('./zeromq-4.1.4.tar.gz'):
        this_dir = pathlib.Path(__file__).parent.absolute()
        return "WGET='curl -O '  ZMQ_URL=file://" + str(this_dir) + "/zeromq-4.1.4.tar.gz "
    else:
        return ""
# https://raw.githubusercontent.com/mli/deps/master/build/zeromq-4.1.4.tar.gz

# wget https://codeload.github.com/openucx/ucx/zip/9229f54 -O ucx.zip
this_dir = pathlib.Path(__file__).parent.absolute()
ucx_path = "file://" + str(this_dir) + "/ucx.zip"
