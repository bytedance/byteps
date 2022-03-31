import timeit
from torchvision import models
import horovod.torch as hvd

class Scheduler():

    def __init__(self, grace, benchmark_step, args):
        self.grace = grace
        self.map_sec = {}
        self.benchmark_step = benchmark_step
        self.epoch = args.scheduler_epoch
        self.step = args.scheduler_step
        self.args = args
        self.warm_up = args.scheduler_warmup
        for _ in range(self.warm_up):
            time = self.get_avg_comm(self.benchmark_step)
        self.search_opt_partition()


    def get_avg_comm(self, benchmark_func):
        ret = 0
        for _ in range(self.epoch):
            ret += benchmark_func()

        return ret / self.epoch


    # get feedback training speed
    def get_speed(self, index):
        if index in self.map_sec:
            return self.map_sec[index]
        self.grace.memory.clean()
        self.grace.compressor.clean()
        self.grace.memory.partition([index])

        # obtain the mean speed of multiple iterations
        comm_time_per_iter = self.get_avg_comm(self.benchmark_step)
        # negotiate the training speed with other workers
        comm_time_per_iter = hvd.broadcast_object(comm_time_per_iter, root_rank=0)

        self.map_sec[index] = comm_time_per_iter
        print("benchmark time: {:.3f} ms\tindex: {}".format(comm_time_per_iter*1000, index))
        return comm_time_per_iter


    #binary search to get the partition with maximum performance
    def search_opt_partition(self):
        tensor_num = self.grace.memory.tensor_num
        start = 1
        end = tensor_num - 1
        # binary search
        while start < end - 1:
            mid = int((start + end) / 2)
            # obtain traing speed under a particular partition
            if self.get_speed(mid) > self.get_speed(mid+self.step):
                start = mid + self.step
            else:
                end = mid

        # get optimal partition
        self.grace.memory.clean()
        self.grace.compressor.clean()
        min_index, min_time = tensor_num//2, float("inf")
        for x in self.map_sec:
            if 1 < x < tensor_num-1 and self.map_sec[x] < min_time:
                min_time = self.map_sec[x]
                min_index = x
        print("final index:\t {}".format(min_index))
        self.grace.memory.partition([min_index])

