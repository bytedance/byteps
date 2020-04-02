import torch
from torch.autograd.function import Function
from byteps.torch.ops import rank, local_rank
from byteps.torch.ops import push_pull_inplace


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
#        count = torch.empty(1,
#                            dtype=running_mean.dtype,
#                            device=input.device).fill_(size)
        count = torch.Tensor([size]).to(input.device)

        my_rank = rank()
        print("rank: ", my_rank, "count: ", count)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

#        count_all = torch.empty(world_size, 1, dtype=count.dtype, device=count.device)
#        mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
#        invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)
#
#        count_l = list(count_all.unbind(0))
#        mean_l = list(mean_all.unbind(0))
#        invstd_l = list(invstd_all.unbind(0))

        # using all_gather instead of all reduce so we can calculate count/mean/var in one go
        # x2682 change this
#        count_all_reduce = torch.distributed.all_gather(count_l, count, process_group, async_op=True)
#        mean_all_reduce = torch.distributed.all_gather(mean_l, mean, process_group, async_op=True)
#        invstd_all_reduce = torch.distributed.all_gather(invstd_l, invstd, process_group, async_op=True)

        # wait on the async communication to finish
#        count_all_reduce.wait()
#        mean_all_reduce.wait()
#        invstd_all_reduce.wait()
        # count_all_reduce, no need to all reduce count, just  compute it
        # for mean and invstd, first zero out, then for rank i fills th i-th
        # element, then do all-reduce, then list(unbind) as above
        count_all = torch.zeros(world_size, 1, dtype=count.dtype, device=count.device)
        mean_all = torch.zeros(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
        invstd_all = torch.zeros(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)
        count_all[my_rank] = size
        mean_all[my_rank] = mean
        invstd_all[my_rank] = invstd

#        print("################################################### about to sync batch norm")
        push_pull_inplace(count_all, average=False, priority=0)
        push_pull_inplace(mean_all, average=False, priority=0)
        push_pull_inplace(invstd_all, average=False, priority=0)

#        count_l = list(count_all.unbind(0))
#        mean_l = list(mean_all.unbind(0))
#        invstd_l = list(invstd_all.unbind(0))
#        print("type(count_all): ", count_all.view(-1))
#        print("rank: ", my_rank, "vvvvvvvvvvvvvvvvvvvvvvv type(count_all): ", count_all)
        print("rank: ", my_rank, "vvvvvvvvvcount_all.view(-1).long().tolist(): ",  count_all.view(-1).long().tolist())
#        tmp = count_all.unbind(1)[0].view(-1).tolist()
#        tmp = tuple(int(val) for val in tmp)
        tmp = count_all.view(-1).tolist()
        tmp = [int(val) for val in tmp]
        print("rank: ", my_rank, "tmp: ", tmp)
        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
#            count_all.view(-1)
#            count_all.view(-1).tolist()
#            tmp
            count_all.view(-1).long().tolist()
        )
        print("rank: ", my_rank, "no probroem with tmp: ", tmp)

#        self.save_for_backward(input, weight, mean, invstd, count_all)
        self.save_for_backward(input, weight, mean, invstd)
        self.process_group = process_group
        self.world_size = world_size

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
#        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        saved_input, weight, mean, invstd = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size

        # calculate local stats as well as grad_weight / grad_bias
#        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel

            # x2682 change this
            # use out-of-place all-reduce:
            # byteps/torch/ops.py:130
            # or in place all-reduce:
            # byteps/torch/ops.py:182

#            sum_dy_all_reduce = torch.distributed.all_reduce(
#                sum_dy, torch.distributed.ReduceOp.SUM, process_group, async_op=True)
#            sum_dy_xmu_all_reduce = torch.distributed.all_reduce(
#                sum_dy_xmu, torch.distributed.ReduceOp.SUM, process_group, async_op=True)

            # wait on the async communication to finish
#            sum_dy_all_reduce.wait()
#            sum_dy_xmu_all_reduce.wait()
            push_pull_inplace(mean_dy, average=False, priority=0)
            push_pull_inplace(mean_dy_xmu, average=False, priority=0)

            mean_dy.div_(world_size)
            mean_dy_xmu.div_(world_size)
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

# x2682 what to do with this class? delete or leave it alone
class CrossMapLRN2d(Function):

    @staticmethod
    def forward(ctx, input, size, alpha=1e-4, beta=0.75, k=1):
        ctx.size = size
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx.scale = None

        assert input.dim() == 4

        ctx.scale = ctx.scale or input.new()
        output = input.new()

        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        output.resize_as_(input)
        ctx.scale.resize_as_(input)

        # use output storage as temporary buffer
        input_square = output
        torch.pow(input, 2, out=input_square)

        pre_pad = int((ctx.size - 1) / 2 + 1)
        pre_pad_crop = channels if pre_pad > channels else pre_pad

        scale_first = ctx.scale.select(1, 0)
        scale_first.zero_()
        # compute first feature map normalization
        for c in range(pre_pad_crop):
            scale_first.add_(input_square.select(1, c))

        # reuse computations for next feature maps normalization
        # by adding the next feature map and removing the previous
        for c in range(1, channels):
            scale_previous = ctx.scale.select(1, c - 1)
            scale_current = ctx.scale.select(1, c)
            scale_current.copy_(scale_previous)
            if c < channels - pre_pad + 1:
                square_next = input_square.select(1, c + pre_pad - 1)
                scale_current.add_(1, square_next)

            if c > pre_pad:
                square_previous = input_square.select(1, c - pre_pad)
                scale_current.add_(-1, square_previous)

        ctx.scale.mul_(ctx.alpha / ctx.size).add_(ctx.k)

        torch.pow(ctx.scale, -ctx.beta, out=output)
        output.mul_(input)

        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = grad_output.new()

        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        paddded_ratio = input.new(channels + ctx.size - 1, input_height,
                                  input_width)
        accum_ratio = input.new(input_height, input_width)

        cache_ratio_value = 2 * ctx.alpha * ctx.beta / ctx.size
        inversePrePad = int(ctx.size - (ctx.size - 1) / 2)

        grad_input.resize_as_(input)
        torch.pow(ctx.scale, -ctx.beta, out=grad_input).mul_(grad_output)

        paddded_ratio.zero_()
        padded_ratio_center = paddded_ratio.narrow(0, inversePrePad,
                                                   channels)
        for n in range(batch_size):
            torch.mul(grad_output[n], output[n], out=padded_ratio_center)
            padded_ratio_center.div_(ctx.scale[n])
            torch.sum(
                paddded_ratio.narrow(0, 0, ctx.size - 1), 0, keepdim=False, out=accum_ratio)
            for c in range(channels):
                accum_ratio.add_(paddded_ratio[c + ctx.size - 1])
                grad_input[n][c].addcmul_(-cache_ratio_value, input[n][c],
                                          accum_ratio)
                accum_ratio.add_(-1, paddded_ratio[c])

        return grad_input, None, None, None, None
