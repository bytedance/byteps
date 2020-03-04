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


class TestMeta(type):
    def __new__(mcls, name, bases, dct):
        test_dtypes = ['float32', 'float16', 'float64'] 

        def gen_test(dtype):
            def test(self):
                return self._run(dtype)
            return test

        for dtype in test_dtypes:
            func_name = "test_local_%s" % dtype
            dct[func_name] = gen_test(dtype)

        return super(TestMeta, mcls).__new__(mcls, name, bases, dct)
