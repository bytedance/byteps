tracker
====

This folder contains tracker scripts that can be used to submit jobs to
different distributed platforms.

(Refactor of https://github.com/dmlc/dmlc-core/tree/master/tracker will merged
back when ready)

## How to use

Assume `prog` is an execuable program, and `[args...]` are the possible
arguments that `prog` accepts.

### Local machine

Run a job using 4 workers and 2 servers on the local machine:

```bash
./dmlc_local.py -s 2 -n 4 ./prog [args...]
```

### Launch via `mpirun`

If `mpirun` is available (shipped with `openmpi` or `mpich2`), we can launch a
job using `dmlc_mpi.py`.
