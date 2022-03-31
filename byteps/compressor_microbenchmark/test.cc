#include <iostream>
#include <omp.h>
#include <chrono>
#include <algorithm> 
#include <cmath>
#include <vector>
#include "efsignSGD.h"
#include "onebit.h"
#include "randomk.h"
#include "topk.h"
#include "dgc.h"

namespace byteps {
namespace common {
namespace compressor {

namespace {
    template<typename type_t>
    void setArray(type_t *array, size_t len) {
    #pragma omp parallel for simd num_threads(64)
        for (int i=0; i<len; i++) {
            float value = rand() / (RAND_MAX + 1.0);
            array[i] = i % 2 ? value : -value;
        }
    }

    enum COMPRESSOR {
        SIGNSGD,
        ONEBIT,
        RANDOMK,
        TOPK,
        DGC
    };

    template <typename T>
    int sum(T* src1, T* src2, size_t len) {
    #pragma omp parallel for simd num_threads(8)
        for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
            src1[i] = src1[i] + src2[i];
        }

        return 0;
    }
}


void compressor_test(size_t size, std::string name, std::vector<std::vector<float>> &results) {
    float *array = new float[size];
    float *residual = new float[size];
    Compressor *compressor;
    if (name == "randomk") {
        compressor = new RandomkCompressor(size*4, BYTEPS_FLOAT32, size, 0.01);
    } else if (name == "dgc") {
        compressor = new DGCCompressor(size*4, BYTEPS_FLOAT32, size, 0.01);
    } else if (name == "efsignSGD") {
        compressor = new EFSignSGDCompressor(size*4, BYTEPS_FLOAT32, size);
    } else if (name == "onebit") {
        compressor = new OnebitCompressor(size*4, BYTEPS_FLOAT32, size);
    } else {
        compressor = new RandomkCompressor(size*4, BYTEPS_FLOAT32, size, 0.01);
    }

    int runs = 10, skip=4;
    long long compress_time = 0, decompress_time = 0;
    for (int i=0; i<runs; i++) {
        setArray<float>(array, size);
        // setArray<float>(residual, size);
        auto start = std::chrono::high_resolution_clock::now();
        sum<float>(array, residual, size*4);
        tensor_t tensor((char*)array, size*4, BYTEPS_FLOAT32);
        tensor_t compressed = compressor->Compress(tensor);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long compress_latency = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (i > skip) {
            compress_time += compress_latency;
        }
        
        start = std::chrono::high_resolution_clock::now();
        compressor->Decompress(compressed);
        elapsed = std::chrono::high_resolution_clock::now() - start;
        long long decompress_latency = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (i > skip) {
            decompress_time += decompress_latency;
        }    
    }

    compress_time /= (runs - skip);
    decompress_time /= (runs - skip);
    float compress_throughput = size*32.0 / compress_time / 1000;
    float decompress_throughput = size*32.0 / decompress_time / 1000;
    results[0].push_back(compress_time);
    results[1].push_back(compress_throughput);
    results[2].push_back(decompress_time);
    results[3].push_back(decompress_throughput);

    // std::cout << "[" << compressor->GetCompressorName() << "] size: " << size 
    // << ", compress time: " << compress_time << " us"
    // << ", compress throughput: " << compress_throughput << " Gbps"
    // << ", decompress time: " << decompress_time << " us"
    // << ", decompress throughput: " << decompress_throughput << " Gbps" 
    // << std::endl;

    delete[] array;
    delete[] residual;
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps



int main(int argc, char *argv[])
{   
    int N = 1024;
    std::vector<std::string> compressor_list{"randomk", "dgc", "efsignSGD", "onebit"};
    for (auto compressor: compressor_list) {
        std::vector<std::vector<float>> results(4, std::vector<float>(0));
        for (int i=0; i<16; i++) {
            std::cout << "size: 2^" << 10+i << std::endl;
            ::byteps::common::compressor::compressor_test((size_t)(pow(2, 10+i)), compressor, results);
        }
        std::cout << compressor << std::endl << "[";
        for (auto result: results) {
            for (auto value: result) {
                std::cout << value << ", ";
            }
            std::cout << "]" << std::endl;
        }

    }

    return 0;
}