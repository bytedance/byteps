#ifndef BYTEPS_COMPRESSOR_IMPL_ONEBIT_H
#define BYTEPS_COMPRESSOR_IMPL_ONEBIT_H

#include "compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Onebit Compressor
 * TODO: update 
 * paper: SIGNSGD: Compressed Optimisation for Non-Convex Problems
 * https://arxiv.org/pdf/1802.04434.pdf
 *
 * each worker i:
 *    c_i <- sign(grad)
 *
 * server: majority vote
 *    sign(\sum_i c_i)
 *
 * \note 0 represents positive and 1 represents negative.
 */
class OnebitCompressor : public Compressor {
 public:
  OnebitCompressor(size_t size, DataType dtype, size_t numel, bool use_scale = true)
      : Compressor(size, dtype, "Onebit"), _numel(numel), _use_scale(use_scale) {
        _num_threads = 4;
        if (std::getenv("OMP_NUM_THREADS")) {
          _num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
        }
      }
  virtual ~OnebitCompressor() = default;

  /*!
   * \brief Compress function
   *
   * compress and pack into byte array.
   * each bit represents a sign.
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  tensor_t Compress(tensor_t grad) override;

  /*!
   * \brief Decompress function
   *
   * unpack from byte array to FP tensor
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  tensor_t Decompress(tensor_t compressed) override;

  /*!
   * \brief help function for error feedback `UpdateError`
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

  template <typename scalar_t, typename index_t>
  tensor_t DecompressImpl(scalar_t* dst, const index_t* src,
                          size_t compressed_size);
                          
  template <typename scalar_t, typename index_t>
  void FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                           const index_t* compressed, size_t compressed_size);

 private:
  int _num_threads;
  size_t _numel;
  bool _use_scale;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_ONEBIT_H