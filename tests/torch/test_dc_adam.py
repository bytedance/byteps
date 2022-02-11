from byteps.torch.ops import dc_adam_
import torch
import byteps.torch as bps
import math
import random
from copy import deepcopy


################################ DC ADAM TEST ###################################
lr = 0.001
weight_decay = 0.0
eps = 1e-5
beta1 = beta2 = 0.5
dc_lambda = 1.0


@torch.no_grad()
def dc_adam(params, grads, prev_params, dc_lambda, exp_avgs, exp_avg_sqs, steps, lr, eps, weight_decay, beta1, beta2):
    for i in range(len(params)):
        p = params[i].clone()
        prev_p = prev_params[i].clone()
        g = grads[i].clone()

        exp_avg_val = exp_avgs[i]
        exp_avg_sq_val = exp_avg_sqs[i]
        step = steps[i]
        
        # update prev_param first before p gets updated
        prev_params[i].copy_(p)

        # dc
        g += dc_lambda * g * g * (p - prev_p)

        # adam
        exp_avg_val = exp_avg_val * beta1 + (1 - beta1) * g
        exp_avg_sq_val = exp_avg_sq_val * beta2 + (1 - beta2) * g * g
        denorm = torch.sqrt(exp_avg_sq_val) + eps
        bias_correction1 = 1 - math.pow(beta1, step)
        bias_correction2 = 1 - math.pow(beta2, step)
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        if weight_decay != 0.0:
            p -= weight_decay * lr * p
        p -= step_size * exp_avg_val / denorm

        params[i].copy_(p)
        grads[i].copy_(g)
        exp_avgs[i].copy_(exp_avg_val)
        exp_avg_sqs[i].copy_(exp_avg_sq_val)



#######################  TEST DC_ADAM GPU ###########################
@torch.no_grad()
def test_dc_adam_fp32():
    print('=' * 20 + "Test DC ADAM on FP32 GPU tensors" + '=' * 20)
    num_tensors = random.randint(2, 100)
    params, grads, prev_params = [], [], []
    exp_avgs, exp_avg_sqs, steps = [], [], []
    max_shape = [1,1]
    for i in range(num_tensors):
        shape = random.sample(range(1, 100), 2)
        if max_shape[0] * max_shape[1] < shape[0] * shape[1]:
            max_shape = shape
        params.append(torch.rand(shape).cuda())
        grads.append(torch.rand(shape).cuda())
        prev_params.append(torch.rand(shape).cuda())
        exp_avgs.append(torch.rand(shape).cuda())
        exp_avg_sqs.append(torch.rand(shape).cuda())
        steps.append(i+1)

    print("Testing setup:")
    print(f"\ttesting on list of {num_tensors} tensors") 
    print(f"\tthe max size of tensors is {max_shape} ")

    # duplicate the parameters
    params_original, prev_params_original = deepcopy(params), deepcopy(prev_params)
    params_cp, grads_cp, prev_params_cp, exp_avgs_cp, exp_avg_sqs_cp = deepcopy(params), deepcopy(grads), deepcopy(prev_params), deepcopy(exp_avgs), deepcopy(exp_avg_sqs)
    assert not (params[0] is params_cp[0])

    # DC ADAM
    dc_adam(params_cp, grads_cp, prev_params_cp, dc_lambda, exp_avgs_cp, exp_avg_sqs_cp, steps, lr, eps, weight_decay, beta1, beta2)
    bps.dc_adam_(params, grads, prev_params, dc_lambda, exp_avgs, exp_avg_sqs, steps, lr, eps, weight_decay, beta1, beta2)


    # test
    ground_truth = [params_original, params_cp, grads_cp, prev_params_cp, exp_avgs_cp, exp_avg_sqs_cp]
    cuda_result = [prev_params, params, grads, prev_params, exp_avgs, exp_avg_sqs]
    names = ["prev_param -- param_original", 
            "params -- params_cp", 
            "grads -- grads_cp", 
            "prev_params -- prev_param_cp", 
            "exp_avgs -- exp_avgs_cp", 
            "exp_avg_sqs -- exp_avg_sqs_cp"]

    print('\nTesting results:')
    print(f"\t{'cuda result -- ground truth'.ljust(30)}: max abs difference")
    print('\t' + "-" * 50)
    for i in range(len(ground_truth)):
        truth, result, name = ground_truth[i], cuda_result[i], names[i]
        max_diff = -math.inf
        for t, r in zip(truth, result):
            max_diff = max(max_diff, torch.max(torch.abs(t - r)).item())
            if not torch.allclose(t, r, atol=1e-3):
                print(f"assert failure: {torch.max(torch.abs(t - r))}")
        print(f"\t{name.ljust(30)}: {max_diff}")
        assert all(torch.allclose(t, r, atol=1e-3) for t, r in zip(truth, result))
    print("\nAll OK!")



#######################  TEST DC_ADAM GPU ###########################
@torch.no_grad()
def test_dc_adam_fp16():
    print('=' * 25 + " Test DC ADAM on FP16 GPU tensors " + '=' * 25)
    num_tensors = random.randint(2, 100)
    params, grads, prev_params = [], [], []
    exp_avgs, exp_avg_sqs, steps = [], [], []
    max_shape = [1,1]
    for i in range(num_tensors):
        shape = random.sample(range(1, 100), 2)
        if max_shape[0] * max_shape[1] < shape[0] * shape[1]:
            max_shape = shape
        params.append(torch.rand(shape).half().cuda())
        grads.append(torch.rand(shape).half().cuda())
        prev_params.append(torch.rand(shape).half().cuda())
        exp_avgs.append(torch.rand(shape).cuda())
        exp_avg_sqs.append(torch.rand(shape).cuda())
        steps.append(i+1)

    # print(params[0])

    print("Testing setup:")
    print(f"\ttesting on list of {num_tensors} tensors") 
    print(f"\tthe max size of tensors is {max_shape} ")
    dc_lambda = 1.0

    # duplicate the parameters
    params_original, prev_params_original = deepcopy(params), deepcopy(prev_params)
    params_cp, grads_cp, prev_params_cp, exp_avgs_cp, exp_avg_sqs_cp = ([p.float() for p in params], 
                                                        [g.float() for g in grads], [p.float() for p in prev_params], 
                                                        [p.clone() for p in exp_avgs], [p.clone() for p in exp_avg_sqs])
    assert params_cp[0].dtype == torch.float32
    assert grads_cp[0].dtype == torch.float32

    # DC ADAM
    dc_adam(params_cp, grads_cp, prev_params_cp, dc_lambda, exp_avgs_cp, exp_avg_sqs_cp, steps, lr, eps, weight_decay, beta1, beta2)
    bps.dc_adam_(params, grads, prev_params, dc_lambda, exp_avgs, exp_avg_sqs, steps, lr, eps, weight_decay, beta1, beta2)

    # print(params[0])
    # print(params_cp[0])
    assert params[0].dtype == torch.float16
    assert params_cp[0].dtype == torch.float32
    assert not torch.isnan(params[0]).any()


    # test
    ground_truth = [params_original, params_cp, grads_cp, prev_params_cp, exp_avgs_cp, exp_avg_sqs_cp]
    cuda_result = [prev_params, params, grads, prev_params, exp_avgs, exp_avg_sqs]
    names = ["prev_param -- param_original", 
            "params -- params_cp", 
            "grads -- grads_cp", 
            "prev_params -- prev_param_cp", 
            "exp_avgs -- exp_avgs_cp", 
            "exp_avg_sqs -- exp_avg_sqs_cp"]

    print('\nTesting results:')
    print(f"\t{'cuda result -- ground truth'.ljust(30)}: max abs difference")
    print('\t' + "-" * 50)
    for i in range(len(ground_truth)):
        truth, result, name = ground_truth[i], cuda_result[i], names[i]
        max_diff = -math.inf
        for t, r in zip(truth, result):
            max_diff = max(max_diff, torch.max(torch.abs(t.float() - r.float())).item())
            if not torch.allclose(t.float(), r.float(), atol=1e-3):
                print(f"assert failure: {torch.max(torch.abs(t.float() - r.float()))}")
        print(f"\t{name.ljust(30)}: {max_diff}")
        assert all(torch.allclose(t.float(), r.float(), atol=1e-3) for t, r in zip(truth, result))
    print("All OK!")



if __name__ == "__main__":
    test_dc_adam_fp32()
    test_dc_adam_fp16()