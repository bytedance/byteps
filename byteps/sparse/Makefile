NVCC=nvcc
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70

NVCCFLAGS = $(NVCCGENCODE) -O3 -std=c++14 --expt-extended-lambda -Xcompiler="-fopenmp,-fPIC" -Wreorder -lineinfo \
	-Igossip/include/cudahelpers/ -I../../3rdparty/ps-lite/include -lrdmacm -libverbs \
	-L../../3rdparty/ps-lite/build -l:libps.a -L../../3rdparty/ps-lite/deps/lib -lzmq
	# -include ../common/cpu_reducer.h -include ../common/common.h -include ../common/logging.h \

# CC = gcc

# CCFLAGS = -O3 -std=c++11 -Wall -fopenmp -fPIC -Wreorder -lineinfo \
# 	-include ../common/cpu_reducer.h \
# 	-Igossip/include/cudahelpers/ -I../../3rdparty/ps-lite/include -lrdmacm -libverbs \
# 	-L../../3rdparty/ps-lite/build -l:libps.a -L../../3rdparty/ps-lite/deps/lib -lzmq -lcudart

GOSSIP_HEADERS = gossip/include/gossip.cuh \
		  gossip/include/gossip/all_to_all_async.cuh \
		  gossip/include/gossip/all_to_all_plan.hpp \
		  gossip/include/gossip/all_to_all.cuh \
		  gossip/include/gossip/broadcast_plan.hpp \
		  gossip/include/gossip/broadcast.cuh \
		  gossip/include/gossip/common.cuh \
		  gossip/include/gossip/context.cuh \
		  gossip/include/gossip/error_checking.hpp \
		  gossip/include/gossip/gather_plan.hpp \
		  gossip/include/gossip/gather.cuh \
		  gossip/include/gossip/memory_manager.cuh \
		  gossip/include/gossip/multisplit.cuh \
		  gossip/include/gossip/point_to_point.cuh \
		  gossip/include/gossip/scatter_plan.hpp \
		  gossip/include/gossip/scatter.cuh \
		  gossip/include/gossip/transfer_plan.hpp 

BUILD_DIR = build

ALL_DEPEND_OBJECTS = $(BUILD_DIR)/plan_parser.o \
					 $(BUILD_DIR)/util.o \
					 $(BUILD_DIR)/common.o \
					 $(BUILD_DIR)/cpu_reducer.o

DEPEND_HEADERS = sparse.h communicator.h common.h util.h loop.h queue_exec.h cpu_reducer.h
				 

.PHONY: all clean

all: sparse

sparse: $(BUILD_DIR) $(BUILD_DIR)/sparse.o $(ALL_DEPEND_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/sparse.o $(ALL_DEPEND_OBJECTS) --shared -o $(BUILD_DIR)/libbyteps.so

$(BUILD_DIR)/sparse.o: $(BUILD_DIR) sparse.cu $(DEPEND_HEADERS) $(ALL_DEPEND_OBJECTS) $(GOSSIP_HEADERS) $(BPS_COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -c sparse.cu -o $(BUILD_DIR)/sparse.o

$(BUILD_DIR)/cpu_reducer.o: cpu_reducer.h cpu_reducer.cc
	$(NVCC) $(NVCCFLAGS) -c cpu_reducer.cc -o $(BUILD_DIR)/cpu_reducer.o

$(BUILD_DIR)/util.o: util.h util.cc
	$(NVCC) $(NVCCFLAGS) -c util.cc -o $(BUILD_DIR)/util.o 

$(BUILD_DIR)/common.o: common.h common.cc
	$(NVCC) $(NVCCFLAGS) -c common.cc -o $(BUILD_DIR)/common.o 

$(BUILD_DIR)/plan_parser.o: gossip/include/plan_parser.cpp gossip/include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) -c gossip/include/plan_parser.cpp -o $(BUILD_DIR)/plan_parser.o

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir $(BUILD_DIR)