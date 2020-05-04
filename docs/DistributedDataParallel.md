# DistributedDataParallel

BytePS Distributed Data Parallel module is compatible with PyTorch Distributed
Data Parallel for the most part. Instead of using PyTorch communication
backends, it uses BytePS push-pull for gradients reduction between nodes.

It currently supports the Single-Process Single-GPU mode. In this mode each
process works with one GPU. Example usage:


```python
# byteps_ddp_example.py
from byteps.torch.parallel import DistributedDataParallel

model = DistributedDataParallel(model, device_ids=[i])
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()
```

Some models have branches, part of the model is skipped during the forward
pass. In that case it's required to call the
DistributedDataParallel.synchronize() function after loss.backward(), e.g.:

```python
# byteps_ddp_example.py
from byteps.torch.parallel import DistributedDataParallel

# construct a model which skips some layers in the forward pass, then wrap the
# model with DistributedDataParallel()
model = DistributedDataParallel(model, device_ids=[i])
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
# the synchronize() call here is required because some layers were skipped in
# the forward pass
model.synchronize()
optimizer.step()
```

To run the program, use `bpslaunch` to launch one process for each device you
wish to use. Refer to the [running](./running.md) document for how to use `bpslaunch`.
