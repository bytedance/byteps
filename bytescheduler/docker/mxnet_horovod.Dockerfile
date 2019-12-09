FROM horovod/horovod:0.16.1-tf1.12.0-torch1.0.0-mxnet1.4.0-py2.7

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_MXNET=1
ENV BYTESCHEDULER_WITHOUT_PYTORCH=1
ENV MXNET_ROOT=/root/incubator-mxnet

WORKDIR /root/

# Clone MXNet as ByteScheduler compilation requires header files
RUN git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git

# Install ByteScheduler
RUN pip install bayesian-optimization
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && python setup.py install

# Examples
WORKDIR /root/bytescheduler/examples/mxnet-image-classification
