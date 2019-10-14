## How to use *byteprofile* for Gluon examples

Users only need to use bps.DistributedTrainer to substitute mx.gluon.Trainer, below shows an example
```python
trainer = bps.DistributedTrainer(params, "sgd", optimizer_params, block=model)
```
Here, `block` must be given to get the dependency info.

Then, users can used the `trainer` to continue training in the same way as they use the original MXNet Gluon API.
