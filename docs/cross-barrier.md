# Cross Global Barrier

This eliminates the global barrier between training iterations for distributed training frameworks (e.g.,
PyTorch), so that the priority-based communication scheduling in BytePS can be effective.

## Why Crossing Barrier?

Existing distributed training frameworks (PyTorch, TensorFlow, etc) do not fully utilize the potentials of overlapping
computation and communication to speed up neural network training: they only support communication overlapping with
backward propagation. But due to layer-wise dependencies in DNN training, we can actually schedule gradient
synchronization order based on when they are consumed in the next iteration, and hence overlap communication with
forward-propagation of the next iteration! Read the paper https://dl.acm.org/citation.cfm?id=3359642 for more
communication scheduling details.

To make this idea work, the first step is to remove the global barrier between two iterations to build layer-wise
dependencies, so that the forward computation of next step can start without waiting for parameter synchronization
completion of all parameters.

Fig.1 shows the dependency graph with global barrier. Machine learning frameworks such as PyTorch and TensorFlow have
similar dependencies when using BytePS for push and pull.

![dag_barrier](https://user-images.githubusercontent.com/13852819/69863244-4b5ee400-12d7-11ea-9356-2dd41dff95ab.png)

*Fig.1: Dependency Graph With Global Barrier*

Fig. 2 shows the dependency graph after removing global barrier. What we do here is to change the dependency
graph from Fig. 1 to Fig. 2 by removing the barrier, building layer-wise dependencies while guaranteeing computation correctness.


![dag_without_barrier](https://user-images.githubusercontent.com/13852819/69863268-5d408700-12d7-11ea-8b39-5e48e3d94c2b.png)
*Fig.2: Dependency Graph After Removing Global Barrier*



