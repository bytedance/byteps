FROM nvidia/cuda:9.0-cudnn7-devel

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_MXNET=1
ENV BYTESCHEDULER_WITHOUT_PYTORCH=1
ENV MXNET_ROOT=/root/incubator-mxnet
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /root

# Install MXNet
RUN apt-get update && apt-get install -y git python-dev build-essential
RUN apt-get install -y wget && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install mxnet-cu90==1.5.0

# Clone MXNet as ByteScheduler compilation needs header files
RUN git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git

# Install ByteScheduler
RUN pip install bayesian-optimization
RUN cd /usr/local/cuda/lib64 && ln -s stubs/libcuda.so libcuda.so.1
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && python setup.py install
RUN rm -f /usr/local/cuda/lib64/libcuda.so.1

# Examples
WORKDIR /root/byteps/bytescheduler/examples/mxnet-image-classification
