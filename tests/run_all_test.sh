#!/bin/bash

TEST_TYPE=onebit ./run_byteps_test.sh
TEST_TYPE=topk ./run_byteps_test.sh
TEST_TYPE=randomk ./run_byteps_test.sh
TEST_TYPE=dithering ./run_byteps_test.sh