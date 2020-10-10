FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG BYTEPS_BASE_PATH=/usr/local
ARG BYTEPS_PATH=$BYTEPS_BASE_PATH/byteps
ARG BYTEPS_GIT_LINK=https://github.com/bytedance/byteps
ARG BYTEPS_BRANCH=master

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        cmake \
        lsb-release \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        python3 \
        python3-dev \
        python3-setuptools

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# install framework
# note: for tf <= 1.14, you need gcc-4.9
ARG FRAMEWORK=tensorflow
RUN if [ "$FRAMEWORK" = "tensorflow" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U tensorflow-gpu==1.15.0; \
    elif [ "$FRAMEWORK" = "pytorch" ]; then \
        pip3 install -U numpy==1.18.1 torchvision==0.5.0 torch==1.4.0; \
    elif [ "$FRAMEWORK" = "mxnet" ]; then \
        pip3 install -U mxnet-cu100==1.5.0; \
    else \
        echo "unknown framework: $FRAMEWORK"; \
        exit 1; \
    fi

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN cd $BYTEPS_BASE_PATH &&\
    git clone --recursive -b $BYTEPS_BRANCH $BYTEPS_GIT_LINK &&\
    cd $BYTEPS_PATH &&\
    python3 setup.py install

ENTRYPOINT ["bpslaunch"]