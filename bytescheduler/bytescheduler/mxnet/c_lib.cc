#include "c_lib.h"

using namespace mxnet;

namespace bytescheduler {
namespace mxnet {

extern "C" int bytescheduler_mxnet_on_complete(void* on_complete) {
    MX_API_BEGIN();

    (*(Callback*)on_complete)();
    delete (Callback*)on_complete;

    MX_API_END();
}

extern "C" int bytescheduler_mxnet_op(NDArray** in_tensor, int in_count,
                                      NDArray** out_tensor, int out_count,
                                      bytescheduler_callback_t callback,
                                      int priority) {
    MX_API_BEGIN();

    std::vector<VarHandle> in_var;
    for (int i = 0; i < in_count; i++) {
        in_var.push_back(in_tensor[i]->var());
    }

    std::vector<VarHandle> out_var;
    for (int i = 0; i < out_count; i++) {
        out_var.push_back(out_tensor[i]->var());
    }

    auto mxnet_op = [callback](RunContext rctx, Callback on_complete) mutable {
        Callback *p = new Callback(on_complete);
        callback((void*)p);
    };

    Engine::Get()->PushAsync(mxnet_op, Context::CPU(),
                             in_var, out_var,
                             FnProperty::kCPUPrioritized, priority, "ByteSchedulerOp");

    MX_API_END();
}

extern "C" int bytescheduler_mxnet_barrier(NDArray** in_tensor, int in_count,
                                           NDArray** out_tensor, int out_count,
                                           int priority) {
    MX_API_BEGIN();

    std::vector<VarHandle> in_var;
    for (int i = 0; i < in_count; i++) {
        in_var.push_back(in_tensor[i]->var());
    }

    std::vector<VarHandle> out_var;
    for (int i = 0; i < out_count; i++) {
        out_var.push_back(out_tensor[i]->var());
    }

    auto mxnet_barrier_op = [](RunContext rctx, Callback on_complete) mutable {
        on_complete();
    };

    Engine::Get()->PushAsync(mxnet_barrier_op, Context::CPU(),
                             in_var, out_var,
                             FnProperty::kCPUPrioritized, priority, "ByteSchedulerBarrier");
    
    MX_API_END();
}

extern "C" int bytescheduler_get_ndarray_avatar(NDArray* tensor,
                                                NDArrayHandle *out) {
    MX_API_BEGIN();

    // a first tensor init triggers segmentation fault at worker side when enabling mkldnn
#if MXNET_USE_MKLDNN == 1
    if (tensor->IsMKLDNNData()) {
        const mkldnn::memory* const_ptr = tensor->GetMKLDNNData();
        mkldnn::memory* raw_ptr = const_cast<mkldnn::memory*>(const_ptr);
        // let shared_ptr do nothing when destroyed
        auto mem_ptr = std::shared_ptr<mkldnn::memory>(raw_ptr, [](mkldnn::memory *mem){});
        // not sure whether we need this
        MKLDNNStream::Get()->RegisterMem(mem_ptr);
        *out = new NDArray(mem_ptr);
    }
    else 
#endif

    {
        auto tlob = tensor->data();        
        auto context = tensor->ctx();
        *out = new NDArray(tlob, context.real_dev_id());
    }

    MX_API_END();

}

} // namespace mxnet
} // namespace bytescheduler