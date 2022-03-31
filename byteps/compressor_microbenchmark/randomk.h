#ifndef BYTEPS_COMPRESSOR_IMPL_RANDOMK_H
#define BYTEPS_COMPRESSOR_IMPL_RANDOMK_H

#include <random>
#include "compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief random number generator based on xorshift128plus
 *
 * refer to https://en.wikipedia.org/wiki/Xorshift#xorshift+
 */
class XorShift128PlusBitShifterRNG {
 public:
  XorShift128PlusBitShifterRNG() {
    std::random_device rd;
    _state = {rd(), rd()};
  }

  // uniform int among [low, high)
  uint64_t Randint(uint64_t low, uint64_t high) {
    return xorshift128p() % (high - low) + low;
  };

  // uniform [0, 1]
  double Rand() { return double(xorshift128p()) / MAX; }

  // Bernoulli Distributation
  bool Bernoulli(double p) { return xorshift128p() < p * MAX; }

  void set_seed(uint64_t seed) { _state = {seed, seed}; }

 private:
  struct xorshift128p_state {
    uint64_t a, b;
  };

  uint64_t xorshift128p() {
    uint64_t t = _state.a;
    uint64_t const s = _state.b;
    _state.a = s;
    t ^= t << 23;        // a
    t ^= t >> 17;        // b
    t ^= s ^ (s >> 26);  // c
    _state.b = t;
    return t + s;
  };

  xorshift128p_state _state;

  static constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();
};


/*!
 * \brief RandomK Compressor
 *
 * paper: Sparsified SGD with Memory
 * https://arxiv.org/pdf/1809.07599.pdf
 *
 * randomly sending k entries of the stochastic gradient
 *
 * \note it is a stochastic algorithm. If you want to have deterministic
 * behavior, please set a seed in the configurations.
 */
class RandomkCompressor : public Compressor {
 public:
  RandomkCompressor(size_t size, DataType dtype, size_t numel, float compress_ratio, unsigned int seed = 0)
      : Compressor(size, dtype, "RandomK"), _numel(numel), _compress_ratio(compress_ratio) {
        _num_threads = 4;
        if (std::getenv("OMP_NUM_THREADS")) {
          _num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
        }
    if (seed != 0) {
      _rng.set_seed(seed);
    }
  };
  virtual ~RandomkCompressor() = default;

  /*!
   * \brief Compress function
   *
   * randomly select k entries and corresponding indices
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  tensor_t Compress(tensor_t grad) override;

  /*!
   * \brief Decompress function
   *
   * fill a zero tensor with topk entries and corresponding indices
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  tensor_t Decompress(tensor_t compressed) override;

  /*!
   * \brief faster version of `UpdateError`
   *
   * 1. e <- p (e is the error and p is the corrected gradient)
   * 2. zero-fill e with selected k indices
   *
   * \param corrected gradient corrected with error
   * \param error error
   * \param compressed compressed gradient
   */
  void FastUpdateError(tensor_t error, tensor_t corrected,
                       tensor_t compressed) override;

 private:
  template <typename index_t, typename scalar_t>
  tensor_t CompressImpl(index_t* dst, const scalar_t* src, size_t len);

  template <typename index_t, typename scalar_t>
  tensor_t DecompressImpl(scalar_t* dst, const index_t* src,
                          size_t compressed_size);

  template <typename index_t, typename scalar_t>
  void FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                           const index_t* compressed, size_t compressed_size);

 private:
  int _num_threads;
  size_t _numel;
  float _compress_ratio;
  XorShift128PlusBitShifterRNG _rng;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_RANDOMK_H