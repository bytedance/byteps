^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for BytePS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
0.2.4 (2020-06)
------------------
* Fix compatibility issue with tf2 + standalone keras
* Add support for tensorflow.keras
* Improve robustness of broadcast


0.2.3 (2020-05)
------------------
* Add DistributedDataParallel module for PyTorch
* Fix the problem of different CPU tensor using the same name
* Add skip_synchronize api for PyTorch
* Add the option for lazy/non-lazy init


0.2.0 (2020-02)
------------------
* Largely improve RDMA performance by enforcing page aligned memory.
* Add IPC support for RDMA. Now support colocating servers and workers without sacrificing much performance.
* Fix a hanging bug in BytePS server.
* Fix RDMA-related segmentation fault problem during fork() (e.g., used by PyTorch data loader).
* New feature: Enable mixing use of colocate and non-colocate servers, along with a smart tensor allocation strategy.
* New feature: Add ``bpslaunch`` as the command to launch tasks.
* Add support for pip install: ``pip3 install byteps``


0.1.0 (2019-12)
------------------
* First official release.
