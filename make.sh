#!/bin/bash

BYTEPS_PATH=$(cd `dirname $0`; pwd)

cd $BYTEPS_PATH
cd 3rdparty/ps-lite/
make clean
make -j USE_RDMA=1

cd $BYTEPS_PATH
mkdir -p 3rdparty/lib
cp 3rdparty/ps-lite/build/libps.a 3rdparty/lib/libps.a

cd $BYTEPS_PATH
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so
ln -sf /usr/local/cuda/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so.1
python setup.py install
rm -f /usr/local/cuda/lib64/libcuda.so
rm -f /usr/local/cuda/lib64/libcuda.so.1


