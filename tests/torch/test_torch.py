# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

import byteps.torch as bps
import itertools
import torch
import os
import numpy as np
import time

has_gpu = os.environ.get('TEST_GPU', '1') == '1'

class TorchTest:
    """
    Tests for ops in byteps.torch
    """

    def _current_context(self):
        if has_gpu:
            torch.cuda.set_device(bps.local_rank())
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def filter_supported_types(self, types):
        return types

    def test_byteps_push_pull(self):
        """Test that the push_pull correctly sums 1D, 2D, 3D tensors."""
        size = bps.size()
        dtypes = self.filter_supported_types([torch.float32])
        dims = [1]
        ctx = self._current_context()

        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.ones((100, 100), device=ctx, dtype=dtype)
            for count in range(20):
                result = bps.push_pull(tensor, name=f"tensor_{count % 10}", average=False)
                result_np = result.cpu().numpy()
                assert np.sum(result_np != size) == 0, result_np
                print(f'test_byteps_push_pull done iter {count}', flush=True)
        print(f'DONE test_byteps_push_pull gpu={has_gpu}', flush=True)


if __name__ == '__main__':
    test = TorchTest()
    bps.init(lazy=False)
    test.test_byteps_push_pull()
    time.sleep(1)