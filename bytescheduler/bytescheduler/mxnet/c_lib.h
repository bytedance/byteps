#ifndef BYTESCHEDULER_MXNET_C_LIB_H
#define BYTESCHEDULER_MXNET_C_LIB_H

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/c_api_error.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/kvstore.h>
//#include <ps/ps.h>

#if MXNET_USE_MKLDNN == 1
#include <mkldnn_base-inl.h>
#endif

namespace bytescheduler {
namespace mxnet {

typedef ::mxnet::Engine Engine;
typedef ::mxnet::NDArray NDArray;
typedef ::mxnet::Engine::CallbackOnComplete Callback;
typedef ::mxnet::Engine::VarHandle VarHandle;
typedef ::mxnet::KVStore KVStore;

typedef void (*bytescheduler_callback_t)(void*);

extern "C" int bytescheduler_mxnet_on_complete(void* on_complete);
extern "C" int bytescheduler_mxnet_op(NDArray** in_tensor, int in_count,
                                      NDArray** out_tensor, int out_count,
                                      bytescheduler_callback_t callback,
                                      int priority);
extern "C" int bytescheduler_mxnet_barrier(NDArray** in_tensor, int in_count,
                                           NDArray** out_tensor, int out_count,
                                           int priority);
extern "C" int bytescheduler_get_ndarray_avatar(NDArray* tensor,
                                                NDArrayHandle *out);
} // namespace mxnet
} // namespace bytescheduler

#endif // BYTESCHEDULER_MXNET_C_LIB_H