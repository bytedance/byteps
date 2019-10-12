# Prebuilt Images

Belows are prebuilt docker images, and their associated source dockerfiles. These prebuilt images might not be up-to-date. 
You may need to manually build them to get the latest functionalities of BytePS, using the source dockerfiles.   

| Docker Image Name | Source Dockerfile | Description |
| --- | --- | --- |
| bytepsimage/worker_mxnet            | Dockerfile.worker.mxnet.cu90            | worker image for MXNet (CUDA 9.0) |
| bytepsimage/worker_pytorch          | Dockerfile.worker.pytorch.cu90          | worker image for PyTorch (CUDA 9.0) |
| bytepsimage/worker_tensorflow       | Dockerfile.worker.tensorflow.cu90       | worker image for TensorFlow (CUDA 9.0) |
| bytepsimage/worker_mxnet_rdma       | Dockerfile.worker.mxnet.cu100.rdma      | worker image for MXNet with RDMA support (CUDA 10.0) |
| bytepsimage/worker_pytorch_rdma     | Dockerfile.worker.pytorch.cu100.rdma    | worker image for PyTorch with RDMA support (CUDA 10.0) |
| bytepsimage/worker_tensorflow_rdma  | Dockerfile.worker.tensorflow.cu100.rdma | worker image for TensorFlow with RDMA support (CUDA 10.0) |
| bytepsimage/byteps_server           | Dockerfile.server                       | server/scheduler image |
| bytepsimage/byteps_server_rdma      | Dockerfile.server.rdma                  | server/scheduler image with RDMA support |
| bytepsimage/mxnet15                 | Dockerfile.mix.mxnet15                  | all-in-one image with MXNet 1.5.0 (CUDA 10.0), applicable to worker/server/scheduler |

