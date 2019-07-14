# ByteScheduler

Bytescheduler eliminates the global barrier between training iterations for distributed training frameworks (e.g., 
PyTorch), so that the priority-based communication scheduling in BytePS can be effective.

## Why ByteScheduler?

Existing distributed training frameworks (MXNet, PyTorch, etc) do not fully utilize the potentials of overlapping 
computation and communication to speed up neural network training: they only support communication overlapping with 
backward propagation. But due to layer-wise dependencies in DNN training, we can actually schedule gradient 
synchronization order based on when they are consumed in the next iteration, and hence overlap communication with 
forward-propagation of the next iteration! Read BytePS's technical report (to be released later) for more communication scheduling details. 
To make this idea work, the first step is to remove the global barrier between two iterations to build layer-wise 
dependencies, so that the forward computation of next step can start without waiting for parameter synchronization 
completion of all parameters. 

Fig.1 shows the dependency graph with global barrier. Machine learning frameworks such as PyTorch and TensorFlow have 
similar dependencies when using BytePS for push and pull.

![](docs/DAG_barrier.png)

*Fig.1: Dependency Graph With Global Barrier*

Fig. 2 shows the dependency graph after removing global barrier. What ByteScheduler does is to change the dependency 
graph from Fig. 1 to Fig. 2 by removing the barrier, building layer-wise dependencies while guaranteeing computation correctness.

![](docs/DAG.png)
*Fig.2: Dependency Graph After Removing Global Barrier*

## Usage

To use ByteScheduler, make a few code additions to your BytePS program. The running command for distributed training
 is same as BytePS. Currently ByteScheduler supports PyTorch with 3 optimizers (SGD, Adam and RMSprop). More support for
 other optimizers and for TensorFlow will be considered later.

### PyTorch

Below shows an example (also see a full benchmark [example](../../example/pytorch/benchmark_bytescheduler.py)). 

```python
# After set up model and optimizer
# ByteScheduler: wrap optimizer using byteps DistributedOptimizer but with two more arguments, i.e., model, num_steps
import byteps.bytescheduler.torch.optimizer as bps
optimizer = bps.DistributedOptimizer(model, optimizer, named_parameters, compressionm, backward_passes_per_step, num_steps)
# Continue
```

## Debugging

To enable debugging mode, set `BYTESCHEDULER_DEBUG=1` and check log in bytescheduler.log. 
Please submit a ticket if you have trouble in using ByteScheduler.



