FROM mxnet/python:nightly_gpu_cu90_py2

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_MXNET=1
ENV BYTESCHEDULER_WITHOUT_PYTORCH=1
ENV MXNET_ROOT=/root/incubator-mxnet

ARG MXNET_VERSION=c6516cc133106f44e247f9ed45165226e3682b62

WORKDIR /root

# Clone MXNet as ByteScheduler compilation requires header files
RUN apt-get install -y git
RUN git clone --recursive https://github.com/apache/incubator-mxnet.git && \
    cd incubator-mxnet && git reset --hard ${MXNET_VERSION}

# Install ByteScheduler
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && python setup.py install

# Examples
WORKDIR /root/bytescheduler/examples/mxnet-image-classification
