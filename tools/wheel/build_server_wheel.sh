#!/bin/bash

# build the docker image
docker build -t byteps.server . -f Dockerfile.server.wheel

# create a docker
id=$(docker create byteps.server)

# copy the wheel from the docker to the host
docker cp $id:/root/mxnet-build/tools/pip/dist/byteps_server-1.5.0-py2.py3-none-manylinux1_x86_64.whl .

# remove the docker
docker rm -v $id

echo successfully built the wheel package

