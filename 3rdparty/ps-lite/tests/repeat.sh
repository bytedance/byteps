#!/bin/bash

set -x

if [ $# -lt 2 ]; then
    echo "usage: $0 num_repeats bin [args...]"
    exit -1;
fi
np=$1
shift

for ((i=0; i<${np}; ++i)); do
    echo "repeat $i: $@"
    $@
    if [ $? != 0 ]; then
        break
    fi
done
