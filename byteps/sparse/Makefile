NVCC=nvcc
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70

NVCCFLAGS = $(NVCCGENCODE) -O3 -std=c++14 --expt-extended-lambda -Xcompiler="-fopenmp,-fPIC" -Wreorder -lineinfo \
	  -Igossip/include/cudahelpers/ -I../../3rdparty/ps-lite/include -lrdmacm -libverbs \
		-L../../3rdparty/ps-lite/build -l:libps.a \
		-L../../3rdparty/ps-lite/deps/lib -lzmq 

HEADERS = gossip/include/gossip.cuh \
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

.PHONY: all clean

all: sparse

sparse: $(BUILD_DIR) $(BUILD_DIR)/plan_parser.o $(BUILD_DIR)/util.o $(BUILD_DIR)/comm.o $(BUILD_DIR)/sparse.o
		$(NVCC) $(NVCCFLAGS) \
				$(BUILD_DIR)/plan_parser.o \
				$(BUILD_DIR)/util.o \
				$(BUILD_DIR)/comm.o \
				$(BUILD_DIR)/sparse.o \
				--shared -o $(BUILD_DIR)/libbyteps.so

$(BUILD_DIR)/sparse.o: $(BUILD_DIR) $(HEADERS) gossip/include/plan_parser.hpp $(BUILD_DIR)/plan_parser.o sparse.h $(BUILD_DIR)/util.o $(BUILD_DIR)/comm.o
	$(NVCC) $(NVCCFLAGS) -c sparse.cu -o $(BUILD_DIR)/sparse.o

$(BUILD_DIR)/util.o: util.h 
	$(NVCC) $(NVCCFLAGS) -c util.cc -o $(BUILD_DIR)/util.o 

$(BUILD_DIR)/comm.o: comm.h 
	$(NVCC) $(NVCCFLAGS) -c comm.cc -o $(BUILD_DIR)/comm.o 

$(BUILD_DIR)/plan_parser.o: gossip/include/plan_parser.cpp gossip/include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) -c gossip/include/plan_parser.cpp  -o $(BUILD_DIR)/plan_parser.o

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir $(BUILD_DIR)