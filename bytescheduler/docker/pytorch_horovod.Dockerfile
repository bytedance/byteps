FROM horovod/horovod:0.16.1-tf1.12.0-torch1.0.0-mxnet1.4.0-py2.7

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_PYTORCH=1
ENV BYTESCHEDULER_WITHOUT_MXNET=1

ARG HOROVOD_VERSION=b5cbf240909b467c348683c69df4d73f07147860

WORKDIR /root/

# Apply the patch and reinstall Horovod
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git
RUN git clone --recursive https://github.com/horovod/horovod.git && \
    cd horovod && git reset --hard ${HOROVOD_VERSION}
RUN cp byteps/bytescheduler/bytescheduler/pytorch/horovod_pytorch.patch horovod/ && \
    cd horovod && git apply horovod_pytorch.patch && python setup.py install

# Install ByteScheduler
RUN pip install bayesian-optimization && cd byteps/bytescheduler && python setup.py install

# Examples
WORKDIR /root/byteps/bytescheduler/examples/
