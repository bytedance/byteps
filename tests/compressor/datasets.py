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

import mxnet as mx


def fake_data(config, dtype, height=224, width=224, depth=3, num_classes=1000):
    mx.random.seed(config.seed)
    image = mx.ndarray.random.normal(-1, 1,
                                     shape=[1, depth, height, width], dtype=dtype)
    label = mx.ndarray.random.randint(0, num_classes, [1, 1])

    images = mx.ndarray.repeat(image, config.data_size, axis=0)
    labels = mx.ndarray.repeat(label, config.data_size, axis=0)

    fake_dataset = mx.gluon.data.ArrayDataset(images, labels)

    return mx.gluon.data.DataLoader(fake_dataset, batch_size=config.batch_size, num_workers=config.num_workers, last_batch='discard')
