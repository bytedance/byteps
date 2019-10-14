## How to use *byteprofile* for Gluon examples

Users only need to use bps.DistributedTrainer to substitute mx.gluon.Trainer, below shows an example
```python
trainer = bps.DistributedTrainer(params, "sgd", optimizer_params,
                                block=model,
                                train_data=train_data, 
                                ctx=context
                                )
```
Here, `block` must be given to get the dependency info, and `train_data` and `ctx` must also be given for warmup training, exporting the model and then importing again using `SymbolBlock.import`.

Next, we can call `bps.DistributedTrainer.update_model` to get the latest model, if `TRACE_ON` is not enabled, just return the original model
```python
model = trainer.update_model()
```
Then, users can used the `model` and `trainer` to continue training in the same way as they use the original MXNet Gluon API.
