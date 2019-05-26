/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_RANGE_H_
#define PS_RANGE_H_
#include "ps/internal/utils.h"
namespace ps {

/**
 * \brief a range [begin, end)
 */
class Range {
 public:
  Range() : Range(0, 0) {}
  Range(uint64_t begin, uint64_t end) : begin_(begin), end_(end) { }

  uint64_t begin() const { return begin_; }
  uint64_t end() const { return end_; }
  uint64_t size() const { return end_ - begin_; }
 private:
  uint64_t begin_;
  uint64_t end_;
};

}  // namespace ps
#endif  // PS_RANGE_H_
