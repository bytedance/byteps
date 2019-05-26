/**
 *  Copyright (c) 2015 by Contributors
 * \file   parallel_kv_match.h
 * \brief  paralle key-value pairs matching
 */
#ifndef PS_INTERNAL_PARALLEL_KV_MATCH_H_
#define PS_INTERNAL_PARALLEL_KV_MATCH_H_
#include <thread>
#include <algorithm>
#include "ps/sarray.h"
#include "ps/internal/assign_op.h"

namespace ps {
namespace  {
/**
 * \brief thread function, internal use
 *
 * \param src_key start of source key
 * \param src_key_end end of source key
 * \param src_val start of source val
 * \param dst_key start of destination key
 * \param dst_key_end end of denstination key
 * \param dst_val start of destination val
 * \param k length of a single value
 * \param op assignment operator
 * \param grainsize thread grainsize size
 * \param n number of matched kv pairs
 */
template <typename K, typename V>
void ParallelOrderedMatch(
    const K* src_key, const K* src_key_end, const V* src_val,
    const K* dst_key, const K* dst_key_end, V* dst_val,
    int k, AsOp op, size_t grainsize, size_t* n) {
  size_t src_len = std::distance(src_key, src_key_end);
  size_t dst_len = std::distance(dst_key, dst_key_end);
  if (dst_len == 0 || src_len == 0) return;

  // drop the unmatched tail of src
  src_key = std::lower_bound(src_key, src_key_end, *dst_key);
  src_val += (src_key - (src_key_end - src_len)) * k;

  if (dst_len <= grainsize) {
    while (dst_key != dst_key_end && src_key != src_key_end) {
      if (*src_key < *dst_key) {
        ++src_key; src_val += k;
      } else {
        if (!(*dst_key < *src_key)) {
          for (int i = 0; i < k; ++i) {
            AssignOp(dst_val[i], src_val[i], op);
          }
          ++src_key; src_val += k;
          *n += k;
        }
        ++dst_key; dst_val += k;
      }
    }
  } else {
    std::thread thr(
        ParallelOrderedMatch<K, V>, src_key, src_key_end, src_val,
        dst_key, dst_key + dst_len / 2, dst_val,
        k, op, grainsize, n);
    size_t m = 0;
    ParallelOrderedMatch<K, V>(
        src_key, src_key_end, src_val,
        dst_key + dst_len / 2, dst_key_end, dst_val + (dst_len / 2) * k,
        k, op, grainsize, &m);
    thr.join();
    *n += m;
  }
}
}  // namespace


/**
 * \brief Merge \a src_val into \a dst_val by matching keys. Keys must be unique
 * and sorted.
 *
 * \code
 * if (dst_key[i] == src_key[j]) {
 *    dst_val[i] op= src_val[j]
 * }
 * \endcode
 *
 * When finished, \a dst_val will have length `k * dst_key.size()` and filled
 * with matched value. Umatched value will be untouched if exists or filled with 0.
 *
 * \tparam K type of key
 * \tparam V type of value
 * \tparam C type of the container such as \ref SArray or \ref std::vector
 * \param src_key the source keys
 * \param src_val the source values
 * \param dst_key the destination keys
 * \param dst_val the destination values.
 * \param k the length of a single value (default is 1)
 * \param op the assignment operator (default is ASSIGN)
 * \param num_threads number of thread (default is 1)
 * \return the number of matched kv pairs
 */
template <typename K, typename V, typename C>
size_t ParallelOrderedMatch(
    const SArray<K>& src_key, const SArray<V>& src_val,
    const SArray<K>& dst_key, C* dst_val,
    int k = 1, AssignOp op = ASSIGN, int num_threads = 1) {
  // do check
  CHECK_GT(num_threads, 0);
  CHECK_EQ(src_key.size() * k, src_val.size());
  CHECK_NOTNULL(dst_val->resize(dst_key.size() * k));
  if (dst_key.empty()) return 0;

  // shorten the matching range
  Range range = FindRange(dst_key, src_key.begin(), src_key.end());
  size_t grainsize = std::max(range.size() * k / num_threads + 5,
                              static_cast<size_t>(1024*1024));
  size_t n = 0;
  ParallelOrderedMatch<K, V>(
      src_key.begin(), src_key.end(), src_val.begin(),
      dst_key.begin() + range.begin(), dst_key.begin() + range.end(),
      dst_val->begin() + range.begin()*k, k, op, grainsize, &n);
  return n;
}

}  // namespace ps
#endif  // PS_INTERNAL_PARALLEL_KV_MATCH_H_
