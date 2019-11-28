# Prebuilt Images

Belows are prebuilt docker images, and their associated source dockerfiles. These prebuilt images might not be up-to-date. 
You may need to manually build them to get the latest functionalities of BytePS, using the source dockerfiles.   

| Docker Image Name | Source Dockerfile | Description |
| --- | --- | --- |
| bytepsimage/mxnet            | Dockerfile.mxnet.cu90            | Image for MXNet (CUDA 9.0) |
| bytepsimage/pytorch          | Dockerfile.pytorch.cu90          | Image for PyTorch (CUDA 9.0) |
| bytepsimage/tensorflow       | Dockerfile.tensorflow.cu90       | Image for TensorFlow (CUDA 9.0) |
| bytepsimage/mxnet15          | Dockerfile.mxnet15.cu100         | Image for MXNet 1.5.0 (CUDA 10.0) |
| bytepsimage/mxnet_rdma       | Dockerfile.mxnet.cu100.rdma      | Image for MXNet with RDMA support (CUDA 10.0) |
| bytepsimage/pytorch_rdma     | Dockerfile.pytorch.cu100.rdma    | Image for PyTorch with RDMA support (CUDA 10.0) |
| bytepsimage/tensorflow_rdma  | Dockerfile.tensorflow.cu100.rdma | Image for TensorFlow with RDMA support (CUDA 10.0) |
