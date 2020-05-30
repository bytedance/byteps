import mxnet as mx
from mxnet import autograd, gluon
import byteps.mxnet as bps
from gluoncv.model_zoo import get_model
import numpy as np


def fake_data(dtype="float32", height=224, width=224, depth=3, num_classes=1000):
    image = mx.ndarray.random.normal(-1, 1,
                                     shape=[1, depth, height, width],
                                     dtype=dtype)
    label = mx.ndarray.random.randint(0, num_classes, [1, 1])

    images = mx.ndarray.repeat(image, 1024, axis=0)
    labels = mx.ndarray.repeat(label, 1024, axis=0)

    fake_dataset = mx.gluon.data.ArrayDataset(images, labels)

    return mx.gluon.data.DataLoader(fake_dataset, batch_size=32, num_workers=2, last_batch='discard')


bps.init()

ctx = mx.gpu(0)
net = get_model("resnet18_v2")
net.initialize(mx.init.Xavier(), ctx=ctx)
params = net.collect_params()

optimizer_params = {'momentum': 0.9, 'wd': 1e-4,
                    'learning_rate': 0.01}

compression_params = {
    "compressor": "onebit",
    "ef": "vanilla",
    "momentum": "nesterov",
    "scaling": True,
}

trainer = bps.DistributedTrainer(
    params, "sgd", optimizer_params, compression_params=compression_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_data = fake_data()

for i, batch in enumerate(train_data):
    data = batch[0].as_in_context(ctx)
    label = batch[1].as_in_context(ctx)

    with autograd.record():
        output = net(data)
        loss = loss_fn(output, label)

    loss.backward()

    for _, param in params.items():
        if param.grad_req != "null":
            x = param._grad[0].asnumpy()
            break

    def onebit(tensor):
        pass

    trainer.step(32)

    for _, param in params.items():
        if param.grad_req != "null":
            y = param._grad[0].asnumpy()
            break

    print(np.allclose(onebit(x), y))
    input()
