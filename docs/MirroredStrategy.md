# MirroredStrategy

The BytePS MirroredStrategy module is compatible with tensorflow
MultiWorkerMirroredStrategy for the most part. Instead of using the builtin
tensorflow collective communication implementation, it uses BytePS push-pull
for gradients reduction between nodes.

It currently supports the Single-Process Single-GPU mode. In this mode each
process works with one GPU. Example usage:


```python
import byteps.tensorflow as bps
from  byteps.tensorflow.distribute import MirroredStrategy

bps.init()
tf.config.experimental.set_visible_devices(gpus[bps.local_rank()], 'GPU')
strategy = MirroredStrategy(devices=["/gpu:0"])

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=100, steps_per_epoch=70)
```
To run the program, use `bpslaunch` to launch one process for each device you
wish to use. Refer to the [running](./running.md) document for how to use
`bpslaunch`.
