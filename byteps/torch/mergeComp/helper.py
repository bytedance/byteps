import sys, os
from .communicator.DDPbackend import DDPBackend

def byteps_from_params(params):
    model = params.get('model', 'none')
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'byteps')
    model_params = params.get('params', 'none')
    ratio = params.get('ratio', 0.01)
    scheduler_file = params.get('scheduler_file', None)
    scheduler_type = params.get('scheduler_type', 0)
    profile = params.get('profile', False)

    if model_params == 'none':
        sys.exit("No model parameters for grace_from_params()")
    momentum = params.get('momentum', 0.9)
    qsgd_quantum = params.get('quantum', 64)

    if comm == "byteps":
        print("[Training Setup] communicator: BytePS")
    else:
        output_string = f"[Compression Setup] compressor: {comp}\n\tmemory: {mem}\n\tcommunicator: {comm}"
        if comp in ("randomk", "dgc"):
            output_string += f"\n\tsparsity ratio: {ratio}"
        print(output_string)

    if comp == 'dgc':
        from .compressor.pooldgc import PoolDgcCompressor
        compressor = PoolDgcCompressor(compress_ratio=ratio)
    elif comp == 'efsignsgd':
        from .compressor.poolefsignsgd import PoolEFSignSGDCompressor
        compressor = PoolEFSignSGDCompressor()
    elif comp == 'fp16':
        from .compressor.poolfp16 import PoolFP16Compressor
        compressor = PoolFP16Compressor()
    elif comp == 'int8':
        from .compressor.poolint8 import PoolInt8Compressor
        compressor = PoolInt8Compressor()
    elif comp == 'none':
        from .compressor.poolnone import PoolNoneCompressor
        compressor = PoolNoneCompressor()
    elif comp == 'onebit':
        from .compressor.poolonebit import PoolOneBitCompressor
        compressor = PoolOneBitCompressor()
    elif comp == 'qsgd':
        from .compressor.poolqsgd import PoolQSGDCompressor
        compressor = PoolQSGDCompressor(quantum_num=qsgd_quantum)
    elif comp == 'randomk':
        from .compressor.poolrandomk import PoolRandomKCompressor
        compressor = PoolRandomKCompressor(compress_ratio=ratio)
    elif comp == 'signsgd':
        from .compressor.poolsignsgd import PoolSignSGDCompressor
        compressor = PoolSignSGDCompressor()
    elif comp == 'signum':
        from .compressor.poolsignum import PoolSignumCompressor
        compressor = PoolSignumCompressor(momentum=momentum)
    elif comp == 'terngrad':
        from .compressor.poolterngrad import PoolTernGradCompressor
        compressor = PoolTernGradCompressor()
    elif comp == 'topk':
        from .compressor.pooltopk import PoolTopKCompressor
        compressor = PoolTopKCompressor(compress_ratio=ratio)
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from .memory.dgc import DgcMemory
        memory = DgcMemory()
    if mem == 'topk':
        from .memory.topk import TopKMemory
        memory = TopKMemory()
    elif mem == 'none':
        from .memory.none import NoneMemory
        memory = NoneMemory()
    elif mem == 'residual':
        from .memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'efsignsgd':
        from .memory.efsignsgd import EFSignSGDMemory
        memory = EFSignSGDMemory()
    else:
        raise NotImplementedError(mem)

    scheduler_threshold = int(os.getenv('BYTEPS_SCHEDULER_THRES', 1024*256))

    ddp = DDPBackend()
    if comm == "espresso":
        from .communicator.pool_bytecomp import ByteComp
        from .compressor.poolfp16 import PoolFP16Compressor
        from .scheduler.scheduler import Scheduler
        scheduler_type = -1
        return ByteComp(PoolFP16Compressor(), compressor, memory, ddp, Scheduler(scheduler_file, scheduler_type, scheduler_threshold))
    elif comm == "byteps":
        from .communicator.pool_bytecomp import ByteComp
        from .compressor.poolfp16 import PoolFP16Compressor
        from .scheduler.scheduler import Scheduler
        scheduler_type = 0
        return ByteComp(PoolFP16Compressor(), compressor, memory, ddp, Scheduler(scheduler_file, scheduler_type, scheduler_threshold))
    elif comm == "byteps-compress":
        from .communicator.pool_bytecomp import ByteComp
        from .compressor.poolfp16 import PoolFP16Compressor
        from .scheduler.scheduler import Scheduler
        scheduler_type = 5
        if model == "resnet101":
            scheduler_threshold = 1024 * 16
        return ByteComp(PoolFP16Compressor(), compressor, memory, ddp, Scheduler(scheduler_file, scheduler_type, scheduler_threshold))
    elif comm == "allgather":
        from .communicator.ddp_allgather import DDPAllgather
        from .compressor.poolfp16 import PoolFP16Compressor
        return DDPAllgather(PoolFP16Compressor(), compressor, memory, ddp, profile)
    elif comm == "fp16":
        from .communicator.ddp_fp16 import DDPFP16
        from .compressor.poolfp16 import PoolFP16Compressor
        return DDPFP16(PoolFP16Compressor(), memory, ddp, profile)
    elif comm == "hitopkcomm":
        from .communicator.ddp_allgather_twolayer import DDPAllgatherTwolayer
        from .compressor.poolfp16 import PoolFP16Compressor
        thresholds = {"bert": 512, "gpt2": 1024*32, "ugatit": 128, "vgg16": 1024*16, "LSTM": 1024*16, "resnet101": 1024*16}
        threshold = 512
        if model in thresholds:
            threshold = thresholds[model]
        return DDPAllgatherTwolayer(PoolFP16Compressor(), compressor, memory, ddp, threshold, profile)
    elif comm == "hipress":
        from .compressor.poolfp16 import PoolFP16Compressor
        if "resnet" in model:
            from .communicator.ddp_hipress_resnet import DDPHiPressResNet
            return DDPHiPressResNet(PoolFP16Compressor(), compressor, memory, ddp, threshold=1024*1024, profile=profile)
        else:
            from .communicator.ddp_hipress import DDPHiPress
            return DDPHiPress(PoolFP16Compressor(), compressor, memory, ddp, threshold=1024*1024*16, profile=profile)
    else:
        raise NotImplementedError(comm)


def add_parser_arguments(parser):
    parser.add_argument("--local_rank", type=int, 
                        help="Local rank. Necessary for the torch.distributed.launch utility.")
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    parser.add_argument('--model-name', type=str, default="",
                        help='model name')                    
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--compress', action='store_true', default=False,
                        help='use gradient compression')
    parser.add_argument('--compressor', type=str, default='efsignsgd',
                        help='compress algorithm')
    parser.add_argument('--compress-ratio', type=float, default=0.01,
                        help='compress ratio for sparsification')
    parser.add_argument('--memory', type=str, default='residual',
                        help='compress algorithm')
    parser.add_argument('--fusion-num', type=int, default=2,
                        help='the number of merged tensors')
    parser.add_argument('--comm', type=str, default='byteps_compress',
                        help='communication for compression')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use Adam optimizer')
    parser.add_argument('--scheduler-file', type=str, default=None,
                        help='the file for scheduling info')
    parser.add_argument('--scheduler-type', type=int, default=0,
                        help='scheduler for all tensors')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile the compression and communication overhead')                  
    return parser


def init_comm(local_rank):
    DDPBackend.init(local_rank)
    return DDPBackend


def wrap_compress_optimizer_named(model, optimizer, args):
    if args.compress:
        """
        compressor: dgc, efsignsgd, fp16, none, onebit, qsgd, randomk, signsgd, signum, terngrad, threshold, topk
        memory: dgc, none, residual, 1bitadam.   Note: 1bitadam is for Adam
        comm: allreduce, allgather, ps
        """
        params = {
            'model': args.model_name,
            'compressor': args.compressor,
            'memory': args.memory,
            'communicator': args.comm,
            'params': model,
            'ratio': args.compress_ratio,
            'scheduler_file': args.scheduler_file,
            'scheduler_type': args.scheduler_type,
            'profile': args.profile
        }

        comp = byteps_from_params(params)
        compress_config = (1, 0)

        import sys
        sys.path.append("../")
        import sparse_optimizer
        optimizer = sparse_optimizer.DistributedOptimizer(optimizer, compression=comp, named_parameters=model, compress_config=compress_config)

        return optimizer
    else:
        import byteps.torch as bps
        # Byteps: (optional) compression algorithm.
        compression = bps.Compression.fp16 if args.fp16 else bps.Compression.none
        optimizer = bps.DistributedOptimizer(optimizer,
                                         named_parameters=model,
                                         compression=compression)
        return optimizer


def wrap_compress_optimizer(model, optimizer, args):
    return wrap_compress_optimizer_named(model.named_parameters(), optimizer, args)