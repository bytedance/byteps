#include <iostream>
#include <omp.h>
#include <chrono>
#include <algorithm> 
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace byteps {
namespace common {

namespace {
    const int ROW_SIZE = 256;
    // OMP_NUM_THREADS = 16, 8, 4, 2
    const int OMP_NUM_THREADS = 16;

    class Array2D {
        size_t _rows;
        size_t _row_size;
        size_t _total_size;
        std::unique_ptr<uint16_t[]> _data;

    public:
        Array2D(size_t rows, size_t row_size)
            : _rows(rows), 
              _row_size(row_size),
              _total_size(rows * row_size),
              _data{new uint16_t[rows * row_size]} {}

        size_t rows() const { return _rows; }
        size_t row_size() const {return _row_size; }
        size_t total_size() const {return _total_size; }
        uint16_t *data() { return _data.get(); }
        void reset() { std::memset(_data.get(), 0, _rows * _row_size * sizeof(uint16_t)); }
        // generate the array with fake randomness
        void random() { 
            auto data = _data.get();
        #pragma omp parallel for simd schedule(static) num_threads(32)
            for (int i=0; i<_rows * _row_size; i++) {
                data[i] = i;
            }
        }

        uint16_t *operator[](size_t row) { 
            return _data.get() + row * _row_size; 
        }
    };



    class SparseSum {
        // the dense tensor
        Array2D _tensor;
    
    public:
        SparseSum(size_t rows, size_t row_size=ROW_SIZE)
            : _tensor(rows, row_size) {}


        void write_row(size_t row, uint16_t *array, size_t row_size) {
            auto pointer = _tensor[row];
            for (auto i=0; i<row_size; i++) {
                // replace add with memory copy
                pointer[i] += array[i];
            }
        } 


        void write_rows(size_t *row_indices, size_t rows, Array2D &array) {
        #pragma omp parallel for simd schedule(static) num_threads(OMP_NUM_THREADS)
            for (auto i=0; i<rows; i++) {
                write_row(row_indices[i], array[i], array.row_size());
            }
        }


        void read_row(size_t row, uint16_t *array, size_t row_size) {
            auto pointer = _tensor[row];
            for (auto i=0; i<row_size; i++) {
                array[i] = pointer[i];
            }
        }


        void read_rows(size_t *row_indices, size_t rows, Array2D &array) {
        #pragma omp parallel for simd schedule(static) num_threads(OMP_NUM_THREADS)
            for (auto i=0; i<rows; i++) {
                read_row(row_indices[i], array[i], array.row_size());
            }
        }

        void reset() { _tensor.reset(); }
        void random() { _tensor.random(); }
    };


    void generate_indices(size_t *indices, size_t num, size_t upper_bound) {
        #pragma omp parallel for simd num_threads(32)
        for (int i=0; i<num; i++) {
            indices[i] = rand() % upper_bound;
        }
        std::sort(indices, indices+num);
    }

}


template<typename type_t>
std::vector<float> CPU_overhead(size_t sparse_rows, size_t dense_rows) {
    int row_size = 256;

    int runs = 40, skip=10;
    long long sparse_to_dense_time = 0, dense_to_sparse_time = 0;
    for (int i=0; i<runs; i++) {
        SparseSum dense_tensor(dense_rows, row_size);
        Array2D sparse_tensor(sparse_rows, row_size);
        sparse_tensor.random();

        size_t *indices = new size_t[sparse_rows];
        generate_indices(indices, sparse_rows, dense_rows);

        auto start = std::chrono::high_resolution_clock::now();
        dense_tensor.write_rows(indices, sparse_rows, sparse_tensor);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long compress_latency = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (i >= skip) {
            sparse_to_dense_time += compress_latency;
        }
        
        start = std::chrono::high_resolution_clock::now();
        dense_tensor.read_rows(indices, sparse_rows, sparse_tensor);
        elapsed = std::chrono::high_resolution_clock::now() - start;
        long long decompress_latency = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (i >= skip) {
            dense_to_sparse_time += decompress_latency;
        }    
    } 

    std::vector<float> rets;
    sparse_to_dense_time /= (runs - skip);
    dense_to_sparse_time /= (runs - skip);
    rets.push_back(sparse_to_dense_time);
    rets.push_back(dense_to_sparse_time);
    std::cout << "sparse to dense time: " << sparse_to_dense_time << "us, dense to sparse time: " << dense_to_sparse_time << "us\n";

    auto sparse_to_dense_tput = sparse_rows * row_size * sizeof(type_t) * 8.0 / sparse_to_dense_time / 1000;
    auto dense_to_sparse_tput = sparse_rows * row_size * sizeof(type_t) * 8.0 / dense_to_sparse_time / 1000;
    rets.push_back(sparse_to_dense_tput);
    rets.push_back(dense_to_sparse_tput);
    std::cout << "sparse to dense throughput: " << sparse_to_dense_tput << "Gbps, dense to sparse time: " << dense_to_sparse_tput << "Gbps\n\n";

    return rets;
}

}  // namespace common
}  // namespace byteps

template<typename type_t>
void print_vector(std::vector<type_t> vec) {
    std::cout << "[";
    for (type_t v: vec) {
        std::cout << v << ", ";
    }
    std::cout << "]\n";
}


int main(int argc, char *argv[])
{   
    // use short (16 bits) to emulate FP16
    size_t dense_tensor_rows = 439926;
    // size_t dense_tensor_rows = 145608;
    std::vector<int> servers_list = {2, 4, 8, 16};
    std::vector<float> densities = {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2};
    int metrics_num = 4;
    for (int servers: servers_list) {
        size_t dense_tensor_rows_each_server = dense_tensor_rows / servers;
        std::vector<std::vector<float>> rets(metrics_num);
        for (float density: densities) {
            size_t sparse_tensor_rows = dense_tensor_rows_each_server * density * servers;
            std::cout << "dense rows: " << dense_tensor_rows
                << ", node: " << servers
                << ", density: " << density
                << ", sparse_tensor_rows: " << sparse_tensor_rows
                << std::endl;
            auto ret = ::byteps::common::CPU_overhead<uint16_t>(sparse_tensor_rows, dense_tensor_rows_each_server);
            
            for (int i=0; i<metrics_num; i++) {
                rets[i].push_back(ret[i]);
            }
        }

        for (int i=0; i<metrics_num; i++) {
            print_vector<float>(rets[i]);
        }
    }
    return 0;
}