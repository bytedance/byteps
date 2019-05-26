/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_SARRAY_H_
#define PS_SARRAY_H_
#include <string.h>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include "ps/internal/utils.h"
#include "ps/range.h"
namespace ps {

/**
 * \brief Shared array
 *
 * A smart array that retains shared ownership. It provides similar
 * functionalities comparing to std::vector, including data(), size(),
 * operator[], resize(), clear(). SArray can be easily constructed from
 * std::vector, such as
 *
 * \code
 * std::vector<int> a(10); SArray<int> b(a);  // copying
 * std::shared_ptr<std::vector<int>> c(new std::vector<int>(10));
 * SArray<int> d(c);  // only pointer copying
 * \endcode
 *
 * SArray is also like a C pointer when copying and assigning, namely
 * both copy are assign are passing by pointers. The memory will be release only
 * if there is no copy exists. It is also can be cast without memory copy, such as
 *
 * \code
 * SArray<int> a(10);
 * SArray<char> b(a);  // now b.size() = 10 * sizeof(int);
 * \endcode
 *
 * \tparam V the value type
 */
template<typename V>
class SArray {
 public:
  /** \brief empty constructor */
  SArray() { }

  /** \brief empty deconstrcutor */
  ~SArray() { }

  /**
   * \brief Create an array with length n with initialized value
   * \param size the length
   * \param val the initial length (0 in default)
   */
  explicit SArray(size_t size, V val = 0) { resize(size, val); }

  /**
   * \brief construct from another SArray.
   *
   * Zero-copy constructor, namely just copy the pointer
   *
   * \tparam W the value type of the source array
   * \param arr the source array
   */
  template <typename W>
  explicit SArray(const SArray<W>& arr) { *this = arr; }

  /**
   * \brief construct from another SArray.
   *
   * Zero-copy constructor, namely just copy the pointer
   *
   * \tparam W the value type of the source array
   * \param arr the source array
   */
  template <typename W> void operator=(const SArray<W>& arr) {
    size_ = arr.size() * sizeof(W) / sizeof(V);
    CHECK_EQ(size_ * sizeof(V), arr.size() * sizeof(W)) << "cannot be divided";
    capacity_ = arr.capacity() * sizeof(W) / sizeof(V);
    ptr_ = std::shared_ptr<V>(arr.ptr(), reinterpret_cast<V*>(arr.data()));
  }

  /**
   * \brief construct from a c-array
   *
   * Zero-copy constructor, namely just copy the pointer
   *
   * \param data the source data
   * \param size the length
   * \param deletable whether or not can call `delete [] data` when the reference
   * count goes 0
   */

  SArray(V* data, size_t size, bool deletable = false) {
    if (deletable) {
      reset(data, size, [](V* data){ delete [] data; });
    } else {
      reset(data, size, [](V* data) { });
    }
  }

  /**
   * \brief copy from a c-array
   *
   * \param data the source data
   * \param size the length
   */
  void CopyFrom(const V* data, size_t size) {
    resize(size);
    memcpy(this->data(), data, size*sizeof(V));
  }

  /**
   * \brief copy from another SArray
   *
   * \param other the source data
   */
  void CopyFrom(const SArray<V>& other) {
    if (this == &other) return;
    CopyFrom(other.data(), other.size());
  }

  /**
   * \brief copy from an iterator
   */
  template <typename ForwardIt>
  void CopyFrom(const ForwardIt& first, const ForwardIt& last) {
    int size = static_cast<int>(std::distance(first, last));
    V* data = new V[size];
    reset(data, size, [](V* data){ delete [] data; });
    auto it = first;
    while (size-- > 0) { *data = *it; ++data; ++it; }
  }

  /**
   * \brief construct from a std::vector, copy the data
   */
  explicit SArray(const std::vector<V>& vec) { CopyFrom(vec.data(), vec.size()); }

  /**
   * \brief construct from a shared std::vector pinter, no data copy
   */
  explicit SArray(const std::shared_ptr<std::vector<V>>& vec) {
    ptr_ = std::shared_ptr<V>(vec, vec->data());
    size_ = vec->size();
    capacity_ = size_;
  }

  /** @brief Copy from a initializer_list */
  template <typename W> SArray(const std::initializer_list<W>& list) {
    CopyFrom(list.begin(), list.end());
  }

  /** @brief Copy from a initializer_list */
  template <typename W> void operator=(const std::initializer_list<W>& list) {
    CopyFrom(list.begin(), list.end());
  }

  /**
   * @brief Reset the current data pointer with a deleter
   */
  template <typename Deleter>
  void reset(V* data, size_t size, Deleter del) {
    size_ = size; capacity_ = size; ptr_.reset(data, del);
  }

  /**
   * @brief Resizes the array to size elements
   *
   * If size <= capacity_, then only change the size. otherwise, append size -
   * current_size entries, and then set new value to val
   */
  void resize(size_t size, V val = 0) {
    size_t cur_n = size_;
    if (capacity_ >= size) {
      size_ = size;
    } else {
      V* new_data = new V[size+5];
      memcpy(new_data, data(), size_*sizeof(V));
      reset(new_data, size, [](V* data){ delete [] data; });
    }
    if (size <= cur_n) return;
    V* p = data() + cur_n;
    if (val == 0) {
      memset(p, 0, (size - cur_n)*sizeof(V));
    } else {
      for (size_t i = 0; i < size - cur_n; ++i) { *p = val; ++p; }
    }
  }

  /**
   * @brief Requests that the capacity be at least enough to contain n elements.
   */
  void reserve(size_t size) {
    if (capacity_ >= size) { return; }
    size_t old_size = size_;
    resize(size);
    size_ = old_size;
  }

  /** @brief release the memory */
  void clear() { reset(nullptr, 0, [](V* data) {}); }


  inline bool empty() const { return size() == 0; }
  inline size_t size() const { return size_; }
  inline size_t capacity() const { return capacity_; }

  inline V* begin() { return data(); }
  inline const V* begin() const { return data(); }
  inline V* end() { return data() + size(); }
  inline const V* end() const { return data() + size(); }

  inline V* data() const { return ptr_.get(); }

  /** \brief get the shared pointer */
  inline std::shared_ptr<V>& ptr() { return ptr_; }
  /** \brief get the const shared pointer */
  inline const std::shared_ptr<V>& ptr() const { return ptr_; }

  inline V back() const { CHECK(!empty()); return data()[size_-1]; }
  inline V front() const { CHECK(!empty()); return data()[0]; }
  inline V& operator[] (int i) { return data()[i]; }
  inline const V& operator[] (int i) const { return data()[i]; }

  inline void push_back(const V& val) {
    if (size_ == capacity_) reserve(size_*2+5);
    data()[size_++] = val;
  }

  void pop_back() { if (size_) --size_; }

  void append(const SArray<V>& arr) {
    if (arr.empty()) return;
    auto orig_size = size_;
    resize(size_ + arr.size());
    memcpy(data()+orig_size, arr.data(), arr.size()*sizeof(V));
  }


  /**
   * @brief Slice a segment, zero-copy
   *
   * @param begin the start index segment
   * @param end the end index segment
   * @return the segment [begin, end)
   */
  SArray<V> segment(size_t begin, size_t end) const {
    CHECK_GE(end, begin); CHECK_LE(end, size());
    SArray<V> ret;
    ret.ptr_ = std::shared_ptr<V>(ptr_, data() + begin);
    ret.size_ = end - begin;
    ret.capacity_ = end - begin;
    return ret;
  }

 private:
  size_t size_ = 0;
  size_t capacity_ = 0;
  std::shared_ptr<V> ptr_;
};


/**
 * \brief Find the index range of a segment of a sorted array such that the
 * entries in this segment is within [lower, upper). Assume
 * array values are ordered.
 *
 * An example
 * \code{cpp}
 * SArray<int> a{1 3 5 7 9};
 * CHECK_EQ(Range(1,3), FindRange(a, 2, 7);
 * \endcode
 *
 * \param arr the source array
 * \param lower the lower bound
 * \param upper the upper bound
 *
 * \return the index range
 */
template<typename V>
Range FindRange(const SArray<V>& arr, V lower, V upper) {
  if (upper <= lower) return Range(0, 0);
  auto lb = std::lower_bound(arr.begin(), arr.end(), lower);
  auto ub = std::lower_bound(arr.begin(), arr.end(), upper);
  return Range(lb - arr.begin(), ub - arr.begin());
}


/*! \brief returns a short debug string */
template <typename V>
inline std::string DebugStr(const V* data, int n, int m = 5) {
  std::stringstream ss;
  ss << "[" << n << "]: ";
  if (n < 2 * m) {
    for (int i = 0; i < n; ++i) ss << data[i] << " ";
  } else {
    for (int i = 0; i < m; ++i) ss << data[i] << " ";
    ss << "... ";
    for (int i = n-m; i < n; ++i) ss << data[i] << " ";
  }
  return ss.str();
}

/**
 * \brief print a debug string
 */
template <typename V>
std::ostream& operator<<(std::ostream& os, const SArray<V>& obj) {
  os << DebugStr(obj.data(), obj.size());
  return os;
}

}  // namespace ps
#endif  // PS_SARRAY_H_
