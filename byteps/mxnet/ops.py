# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load all the necessary MXNet C types.
import ctypes
import os
import warnings

import mxnet as mx
from mxnet.base import c_str, check_call, string_types

from byteps.common import get_ext_suffix
from byteps.common import BytePSBasics as _BytePSBasics
_basics = _BytePSBasics(__file__, 'c_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
suspend = _basics.suspend
resume = _basics.resume
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank

dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
MXNET_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)


def byteps_push_pull(tensor, version=0, priority=0, name=None, is_average=True):
    """
    A function that performs pushing and pulling tensors

    The operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all BytePS processes for a given name. The reduction will not
    start until all processes are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires tensors, then callings this function will allow tensors
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.

    Returns:
        None
    """

    c_in = tensor.handle
    if isinstance(name, string_types):
        check_call(MXNET_LIB_CTYPES.byteps_mxnet_push_pull_async(c_in,
                                                                 c_str(name), ctypes.c_int(version), ctypes.c_int(priority), ctypes.c_bool(is_average)))
    else:
        check_call(MXNET_LIB_CTYPES.byteps_mxnet_push_pull_async(c_in,
                                                                 name, ctypes.c_int(version), ctypes.c_int(priority), ctypes.c_bool(is_average)))

    return


def byteps_declare_tensor(name, **kwargs):
    """create ctx for tensors and register compressor 

    Warpper of the c++ function. Build parameter dict.

    Arguments:
        name : str, tensor name
        **kwargs: extra params w.r.t gradient compression  

    Returns:
        None
    """
    def _create_c_style_string_array(strings):
        byte_arr = [bytes(string, 'utf-8') for string in strings]
        arr = (ctypes.c_char_p*len(byte_arr))()
        arr[:] = byte_arr
        return arr

    args = {}
    for k, v in kwargs.items():
        splits = k.split('_')
        if len(splits) < 2 and not all(splits):
            warnings.warn("Ignore invalid params %s of %s" % (k, name))
            continue
        
        # remove first prefix "byteps"
        k = '_'.join(splits[1:])
        if isinstance(v, str):
            args[k] = v.lower()
        elif isinstance(v, (int, float,)):
            args[k] = str(v)
        elif isinstance(v, bool):
            args[k] = str(int(v)).lower()
        else:
            raise ValueError("Invalid %s of type %s of %s" %
                             (v, type(v), name))

    check_call(MXNET_LIB_CTYPES.byteps_mxnet_declare_tensor(
        c_str(name),
        ctypes.c_int(len(args)),
        _create_c_style_string_array(list(args.keys())),
        _create_c_style_string_array(list(args.values()))
    ))