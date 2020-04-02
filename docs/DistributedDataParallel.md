BytePS Distributed Data Parallel module is compatible with PyTorch Distributed
Data Parallel for the most part. Instead of using PyTorch communication
backends, it uses BytePS push-pull for gradients reduction between nodes.

It currently supports the Single-Process Single-GPU mode. In this mode each
process works with one GPU. Example usage:


```python
# ddp_example.py
from byteps.torch.parallel import DistributedDataParallel

model = DistributedDataParallel(model, device_ids=[i])
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
model.synchronize()
optimizer.step()
```

Some models have branches, part of the model is skipped during the forward
pass. In that case it's required to call the
DistributedDataParallel.synchronize() after loss.backward(), e.g:

```python
# ddp_example.py
from byteps.torch.parallel import DistributedDataParallel

model = DistributedDataParallel(model, device_ids=[i])
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
model.synchronize()
optimizer.step()
```

To run the program, use bpslaunch to launch one process for each device you
wish to use. Refer to the [running](./running.md) document for how to use bpslaunch.
