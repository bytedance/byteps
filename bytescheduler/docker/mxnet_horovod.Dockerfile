FROM horovod/horovod:0.16.1-tf1.12.0-torch1.0.0-mxnet1.4.0-py2.7

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_MXNET=1
ENV BYTESCHEDULER_WITHOUT_PYTORCH=1
ENV MXNET_ROOT=/root/incubator-mxnet

WORKDIR /root/

# Install gcc 4.9
RUN mkdir -p /root/gcc/ && cd /root/gcc &&\
    wget http://launchpadlibrarian.net/247707088/libmpfr4_3.1.4-1_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728424/libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728426/libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728314/gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728399/cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728404/gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728432/libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728401/g++-4.9_4.9.3-13ubuntu2_amd64.deb

RUN cd /root/gcc &&\
    dpkg -i gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libmpfr4_3.1.4-1_amd64.deb &&\
    dpkg -i libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i g++-4.9_4.9.3-13ubuntu2_amd64.deb

# Pin GCC to 4.9 (priority 200) to compile correctly against MXNet
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

# May need to uninstall default MXNet and install mxnet-cu90==1.5.0

# Clone MXNet as ByteScheduler compilation requires header files
RUN git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git
RUN cd incubator-mxnet && git reset --hard 75a9e187d00a8b7ebc71412a02ed0e3ae489d91f

# Install ByteScheduler
RUN pip install bayesian-optimization
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && python setup.py install

# Examples
WORKDIR /root/byteps/bytescheduler/examples/mxnet-image-classification
