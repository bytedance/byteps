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

import unittest

from parameterized import parameterized

import byteps.mxnet as bps


def worker(model, input_data, batch_size):
    if model is None:
        raise ValueError("model is None")


class OnebitCaseBase(unittest.TestCase):
    def _model(self):
        return None

    def _config(self):
        return {
            "batch_size": 64,
            "dtype": "float32",
            "epochs": 5,
            "cpu_num": 2,
            "lr": 0.01,
            "momentum": 0.9,
            "gpu", True
        }

    @parameterized.expand([(1,), (3,), (5)])
    def test_locally(self, num_workers):
        print("it works")


if __name__ == "__main__":
    unittest.main()
