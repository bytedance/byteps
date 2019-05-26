# How To

## Debug PS-Lite

One way to debug is loggining all communications. We can do it by specifying
the environment variable `PS_VERBOSE`:
- `PS_VERBOSE=1`: logging connection information
- `PS_VERBOSE=2`: logging all data communication information

For example, first run `make test; cd tests` in the root directory. Then
```bash
export PS_VERBOSE=1; ./local.sh 1 1 ./test_connection
```
Possible outputs are
```bash
[19:57:18] src/van.cc:72: Node Info: role=schedulerid=1, ip=127.0.0.1, port=8000
[19:57:18] src/van.cc:72: Node Info: role=worker, ip=128.2.211.110, port=58442
[19:57:18] src/van.cc:72: Node Info: role=server, ip=128.2.211.110, port=40112
[19:57:18] src/van.cc:336: assign rank=8 to node role=server, ip=128.2.211.110, port=40112
[19:57:18] src/van.cc:336: assign rank=9 to node role=worker, ip=128.2.211.110, port=58442
[19:57:18] src/van.cc:347: the scheduler is connected to 1 workers and 1 servers
[19:57:18] src/van.cc:354: S[8] is connected to others
[19:57:18] src/van.cc:354: W[9] is connected to others
[19:57:18] src/van.cc:296: H[1] is stopped
[19:57:18] src/van.cc:296: S[8] is stopped
[19:57:18] src/van.cc:296: W[9] is stopped
```
where `H`, `S` and `W` stand for scheduler, server, and worker respectively.

## Use a Particular Network Interface

In default PS-Lite automatically chooses an available network interface. But for
machines have multiple interfaces, we can specify the network interface to use
by the environment variable `DMLC_INTERFACE`. For example, to use the
infinite-band interface `ib0`, we can
```bash
export DMLC_INTERFACE=ib0; commands_to_run
```

If all PS-Lite nodes run in the same machine, we can set `DMLC_LOCAL` to use
memory copy rather than the local network interface, which may improve the
performance:
```bash
export DMLC_LOCAL=1; commands_to_run
```

## Environment Variables to Start PS-Lite

This section is useful if we want to port PS-Lite to other cluster resource
managers besides the provided ones such as `ssh`, `mpirun`, `yarn` and `sge`.

To start a PS-Lite node, we need to give proper values to the following
environment variables.
- `DMLC_NUM_WORKER` : the number of workers
- `DMLC_NUM_SERVER` : the number of servers
- `DMLC_ROLE` : the role of the current node, can be `worker`, `server`, or `scheduler`
- `DMLC_PS_ROOT_URI` : the ip or hostname of the scheduler node
- `DMLC_PS_ROOT_PORT` : the port that the scheduler node is listening

## Retransmission for Unreliable Network

It's not uncommon that a message disappear when sending from one node to another
node. The program hangs when a critical message is not delivered
successfully. In that case, we can let PS-Lite send an additional ACK for each
message, and resend that message if the ACK is not received within a given
time. To enable this feature, we can set the environment variables

- `PS_RESEND` : if or not enable retransmission. Default is 0.
- `PS_RESEND_TIMEOUT` : timeout in millisecond if an ACK message if not
  received. PS-Lite then will resend that message. Default is 1000.

We can set `PS_DROP_MSG`, the percent of probability to drop a received
message, for testing. For example, `PS_DROP_MSG=10` will let a node drop a
received message with 10% probability.
