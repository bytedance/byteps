#!/bin/bash
# main script of travis

if [ ${TASK} == "lint" ]; then
    make lint || exit -1
fi

if [ ${TASK} == "build" ]; then
    make DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit -1
fi

if [ ${TASK} == "test" ]; then
    make test DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit -1
    cd tests
    # single-worker tests
    tests=( test_connection test_kv_app test_simple_app )
    for test in "${tests[@]}"
    do
        find $test -type f -executable -exec ./repeat.sh 4 ./local.sh 2 2 ./{} \;
    done
    # multi-workers test
    multi_workers_tests=( test_kv_app_multi_workers )
    for test in "${multi_workers_tests[@]}"
    do
        find $test -type f -executable -exec ./repeat.sh 4 ./local_multi_workers.sh 2 2 ./{} \;
    done
fi
