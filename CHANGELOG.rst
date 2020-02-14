^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for BytePS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2020-02)
------------------
* Largely improve RDMA performance by enforcing page aligned memory and adding IPC support. Now support colocating servers and workers without sacrificing much performance.
* Fix a hanging bug in BytePS server.
* Fix RDMA fork problem caused by multi-processing (e.g., data loaders).
* New feature: Enable mixing use of colocate and non-colocate servers, along with a smart tensor allocation strategy.
* Add support for pip install.


0.1.0 (2019-12)
------------------
* First official release.