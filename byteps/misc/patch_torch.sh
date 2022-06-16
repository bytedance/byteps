#!/bin/bash

torch_path=$(pip3 show torch | awk '/Location/ {print $2}')/torch
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
byteps_path=${THIS_DIR}/../../byteps

echo "patching torch installation at: ${torch_path}"

###############################################################################
#                                                                             #
torch_dist_init_file=${torch_path}/distributed/__init__.py
if [[ ! -f "${torch_dist_init_file}-bk" ]]; then
sudo cp   ${torch_dist_init_file} "${torch_dist_init_file}-bk"
sudo tee -a ${torch_dist_init_file} <<- 'EOF'

################# begin byteps section

import os
if os.getenv("TORCH_BYTECCL", "O3") in ["O3"]:
    from .byteps_dist_init import init_process_group, broadcast_object_list, all_gather_object

################# end byteps section
EOF
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
# copy byteps/torch/distributed/byteps_dist_init.py and
# byteps/torch/distributed/ProcessGroupBYTEPS.py to the right location. do all
# this in setup.py
sudo cp $byteps_path/torch/distributed/byteps_dist_init.py $torch_path/distributed/byteps_dist_init.py
sudo cp $byteps_path/torch/distributed/ProcessGroupBYTEPS.py $torch_path/distributed/
fi
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
torch_nn_parallel_init_file=${torch_path}/nn/parallel/__init__.py
if [[ ! -f "${torch_nn_parallel_init_file}-bk" ]]; then
sudo cp   ${torch_nn_parallel_init_file} "${torch_nn_parallel_init_file}-bk"
sudo tee -a ${torch_nn_parallel_init_file} <<- 'EOF'

################# begin byteps section

import os
if os.getenv("TORCH_BYTECCL", "O3") in ["O3"]:
    def ByteCCLDistributedDataParallel(*args, **kwargs):
        ddp = torch.nn.parallel.DistributedDataParallel
        return ddp(*args, **kwargs)

    DistributedDataParallel = ByteCCLDistributedDataParallel
    from . import distributed
    distributed.DistributedDataParallel = ByteCCLDistributedDataParallel

################# end byteps section
EOF
fi
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
torch_nn_init_file=${torch_path}/nn/__init__.py
if [[ ! -f "${torch_nn_init_file}-bk" ]]; then
sudo cp   ${torch_nn_init_file} "${torch_nn_init_file}-bk"
sudo tee -a ${torch_nn_init_file} <<- 'EOF'

################# begin byteps section

import os
if os.getenv("TORCH_BYTECCL", "O3") in ["O3"]:
    def ByteCCLSyncBatchNorm(*args, **kwargs):
        syncbn = torch.nn.SyncBatchNorm
        return syncbn(*args, **kwargs)
    SyncBatchNorm = ByteCCLSyncBatchNorm

################# end byteps section
EOF
fi
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
torch_init_file=${torch_path}/__init__.py
if [[ ! -f "${torch_init_file}-bk" ]]; then
sudo cp   ${torch_init_file} "${torch_init_file}-bk"
sudo tee -a ${torch_init_file} <<- 'EOF'

################# begin byteps section

import os
if os.getenv("TORCH_BYTECCL", "O3") in ["O3"]:
    from byteps.torch import parallel as bpsparallel
    bps_ddp = bpsparallel.DistributedDataParallel
    torch.nn.parallel.DistributedDataParallel = bps_ddp
    torch.nn.parallel.distributed.DistributedDataParallel = bps_ddp

    from byteps.torch.sync_batch_norm import SyncBatchNorm as bps_SyncBatchNorm
    torch.nn.SyncBatchNorm = bps_SyncBatchNorm

################# end byteps section
EOF
fi
#                                                                             #
###############################################################################
