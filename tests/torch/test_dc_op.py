import torch
import byteps.torch as bps
import random


@torch.no_grad()
def dc_cpu(params, grads, prev_params, dc_lambda):
    res = []
    for p, g, pp in zip(params, grads, prev_params):
        res.append(g + dc_lambda * g * g * (p - pp))
    return res

##########################################################################
def test_dc_cpu_fp32():
    print('=' * 20 + "Test DC on FP32 CPU tensors" + '=' * 20)
    num_tensors = random.randint(2, 100)
    print(f"testing on list of {num_tensors} tensors")
    params, grads, prev_params = [], [], []
    max_shape = [1,1]
    for _ in range(num_tensors):
        shape = random.sample(range(1, 100), 2)
        if max_shape[0] * max_shape[1] < shape[0] * shape[1]:
            max_shape = shape
        params.append(torch.rand(shape))
        grads.append(torch.rand(shape))
        prev_params.append(torch.rand(shape))
    print(f"the max size of tensors is {max_shape} ")
    dc_lambda = torch.rand(1).item()

    params_cp, grads_cp, prev_params_cp = params.copy(), grads.copy(), prev_params.copy()
    res = dc_cpu(params, grads, prev_params, dc_lambda)
    bps.delay_compensation_(params, grads, prev_params, dc_lambda)

    max_diff = 0
    for r, g in zip(res, grads):
        max_diff = max(max_diff, torch.max(torch.abs(r - g)))
    print(f"max abs difference is {max_diff}")

    assert all(torch.all(p == pp) for p, pp in zip(params, prev_params))
    assert all(torch.all(g == r) for g, r in zip(grads, res))
    assert all(torch.all(p == pc) for p, pc in zip(params, params_cp))
    print("All OK!")


##########################################################################
def test_dc_gpu_fp32():
    print('='*20 + "Test DC on FP32 GPU tensors" + '='*20)
    num_tensors = random.randint(2, 100)
    print(f"testing on list of {num_tensors} tensors")
    params, grads, prev_params = [], [], []
    max_shape = [1, 1]
    for _ in range(num_tensors):
        shape = random.sample(range(1, 100), 2)
        if max_shape[0] * max_shape[1] < shape[0] * shape[1]:
            max_shape = shape
        params.append(torch.rand(shape).cuda())
        grads.append(torch.rand(shape).cuda())
        prev_params.append(torch.rand(shape).cuda())
    print(f"the max size of tensors is {max_shape} ")
    dc_lambda = torch.rand(1).item()

    params_cp = params.copy()
    res = dc_cpu(params, grads, prev_params, dc_lambda)
    bps.delay_compensation_(params, grads, prev_params, dc_lambda)

    max_diff = 0
    for r, g in zip(res, grads):
        max_diff = max(max_diff, torch.max(torch.abs(r - g)))
    print(f"max abs difference is {max_diff}")

    assert all(torch.allclose(p, pp) for p, pp in zip(params, prev_params)), "params should be copied to prev_params"
    assert all(torch.allclose(g, r) for g, r in zip(grads, res)), "grads are not correct"
    assert all(torch.allclose(p, pc) for p, pc in zip(params, params_cp)), "params should be not be touched"
    print("All OK!")



######################################################################
def test_dc_gpu_fp16():
    print('='*20 + "Test DC on FP16 GPU tensors" + '='*20)
    num_tensors = random.randint(2, 100)
    print(f"testing on list of {num_tensors} tensors")
    params, grads, prev_params = [], [], []
    max_shape = [1, 1]
    for _ in range(num_tensors):
        shape = random.sample(range(1, 100), 2)
        if max_shape[0] * max_shape[1] < shape[0] * shape[1]:
            max_shape = shape
        params.append(torch.rand(shape).cuda().half())
        grads.append(torch.rand(shape).cuda().half())
        prev_params.append(torch.rand(shape).cuda().half())
    print(f"the max size of tensors is {max_shape} ")
    dc_lambda = torch.rand(1).item()

    params_single, grads_single, prev_params_single = [p.float() for p in params], [g.float() for g in grads], [pp.float() for pp in prev_params]

    # ground truth use FP32 to compute result
    res = dc_cpu(params_single, grads_single, prev_params_single, dc_lambda)
    res = [r.half() for r in res]

    bps.delay_compensation_(params, grads, prev_params, dc_lambda)

    max_diff = 0
    for r, g in zip(res, grads):
        max_diff = max(max_diff, torch.max(torch.abs(r - g)))
    print(f"max abs difference is {max_diff}")

    assert all( torch.allclose(p, pp) for p, pp in zip(params, prev_params) )
    assert all( torch.allclose(p.float(), ps) for p, ps in zip(params, params_single) )
    assert all( torch.allclose(g, r, atol=1e-3, rtol=1e-3) for g, r in zip(grads, res) )
    print("All OK!")

if __name__ == "__main__":
    test_dc_cpu_fp32()
    test_dc_gpu_fp32()
    test_dc_gpu_fp16()

