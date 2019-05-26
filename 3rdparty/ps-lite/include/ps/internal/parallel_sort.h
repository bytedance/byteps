/**
 *  Copyright (c) 2015 by Contributors
 * @file   parallel_sort.h
 * @brief  Parallel sort
 */
#ifndef PS_INTERNAL_PARALLEL_SORT_H_
#define PS_INTERNAL_PARALLEL_SORT_H_
#include <functional>
#include <thread>
#include <algorithm>
#include "ps/sarray.h"
namespace ps {

namespace  {
/**
 * \brief the thread function
 *
 * \param data start pointer of data
 * \param len  length of data
 * \param grainsize max data length of one thread
 * \param cmp comparison function
 */
template<typename T, class Fn>
void ParallelSort(T* data, size_t len, size_t grainsize, const Fn& cmp) {
  if (len <= grainsize) {
    std::sort(data, data + len, cmp);
  } else {
    std::thread thr(ParallelSort<T, Fn>, data, len/2, grainsize, cmp);
    ParallelSort(data + len/2, len - len/2, grainsize, cmp);
    thr.join();

    std::inplace_merge(data, data + len/2, data + len, cmp);
  }
}
}  // namespace

/**
 * \brief Parallel Sort
 *
 * \param arr the array for sorting
 * \param num_threads number of thread
 * \param cmp the comparision function such as
 * [](const T& a, const T& b) {* return a < b; }
 * or an even simplier version:
 * std::less<T>()
 */
template<typename T, class Fn>
void ParallelSort(SArray<T>* arr,
                  int num_threads = 2,
                  const Fn& cmp = std::less<T>()) {
  CHECK_GT(num_threads, 0);
  CHECK(cmp);
  size_t grainsize = std::max(arr->size() / num_threads + 5, (size_t)1024*16);
  ParallelSort(arr->data(), arr->size(), grainsize, cmp);
}

}  // namespace ps
#endif  // PS_INTERNAL_PARALLEL_SORT_H_
