import mxnet as mx
import mxnet.ndarray as nd
import numpy as np


def fake_data(dtype="float32", batch_size=32, height=224, width=224, depth=3, num_classes=1000):
    image_list = []
    label_list = []
    for _ in range(8):
        image = mx.ndarray.random.normal(-1, 1,
                                         shape=[1, depth, height, width],
                                         dtype=dtype)
        label = mx.ndarray.random.randint(0, num_classes, [1, 1])

        images = mx.ndarray.repeat(image, 128, axis=0)
        labels = mx.ndarray.repeat(label, 128, axis=0)
        # print(labels)
        image_list.append(images)
        label_list.append(labels)

    images = nd.concat(*image_list, dim=0)
    labels = nd.concat(*label_list, dim=0)
    # print(labels)
    fake_dataset = mx.gluon.data.ArrayDataset(images, labels)

    return mx.gluon.data.DataLoader(fake_dataset, batch_size=batch_size, num_workers=4,
                                    shuffle=True, last_batch='discard')


class XorShift128PlusBitShifterRNG:
    def __init__(self, a, b):
        self.state = np.array([a, b], dtype=np.uint64)

    def xorshift128p(self):
        t = self.state[0]
        s = self.state[1]
        self.state[0] = s
        t ^= t << np.uint64(23)
        t ^= t >> np.uint64(17)
        t ^= s ^ (s >> np.uint64(26))
        self.state[1] = t
        return int(t + s)

    def randint(self, low, high):
        return self.xorshift128p() % (high - low) + low
