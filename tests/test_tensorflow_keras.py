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

"""Tests for byteps.keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import warnings

from distutils.version import LooseVersion
from tensorflow import keras
from tensorflow.python.keras import backend as K

import byteps.tensorflow.keras as bps

class TfKerasTests:
    """
    Tests for ops in byteps.keras.
    """

    def __init__(self, *args, **kwargs):
        super(TfKerasTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        bps.init()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.visible_device_list = str(bps.local_rank())

    def test_train_model(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = keras.optimizers.RMSprop(lr=0.0001)
            opt = bps.DistributedOptimizer(opt)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(2, input_shape=(3,)))
            model.add(keras.layers.RepeatVector(3))
            model.add(keras.layers.ThresholdedReLU(0.5))
            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=opt,
                          metrics=[keras.metrics.categorical_accuracy],
                          sample_weight_mode='temporal')

            x = np.random.random((1, 3))
            y = np.random.random((1, 3, 3))

            def generator():
                while 1:
                    yield (x, y)

            print ('x is: ', x)
            print ('y is: ', y)
            # No assertions, we just need to verify that it doesn't hang
            callbacks = [bps.callbacks.BroadcastGlobalVariablesCallback(0)]
            model.fit_generator(generator(),
                                steps_per_epoch=10,
                                callbacks=callbacks,
                                epochs=0,
                                verbose=0,
                                workers=4,
                                initial_epoch=1)
            print ('x-trained is: ', x)
            print ('y-trained is: ', y)

    def test_sparse_as_dense(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = keras.optimizers.RMSprop(lr=0.0001)
            opt = bps.DistributedOptimizer(opt, sparse_as_dense=True)

            model = keras.models.Sequential()
            model.add(keras.layers.Embedding(1000, 64, input_length=10))
            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=opt)

            x = np.random.randint(1000, size=(32, 10))
            y = np.random.random((32, 10, 64))
            # No assertions, we just need to verify that it doesn't hang
            model.train_on_batch(x, y)


if __name__ == '__main__':
    keras_test = TfKerasTests()
    keras_test.test_train_model()
