# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2019 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright 2021 Bytedance Inc. or its affiliates. All Rights Reserved.
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

from distutils.version import LooseVersion

import os
import platform
import sys
import unittest
import warnings
import time

import torch
import byteps.torch as bps

class TorchTests(unittest.TestCase):
    """
    Tests for ops in byteps.torch.
    """

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_broadcast_object(self):
        expected_obj = {
            'hello': 123,
            0: [1, 2]
        }
        obj = expected_obj if bps.rank() == 0 else {}

        obj = bps.broadcast_object(obj, root_rank=0)
        self.assertDictEqual(obj, expected_obj)

if __name__ == '__main__':
    test = TorchTests()
    bps.init(lazy=False)
    time.sleep(2)
    test.test_broadcast_object()
