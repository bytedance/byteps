import sys
import byteps.tensorflow as bps


def byteps_from_params(params):
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allgather')
    model_params = params.get('params', 'none')
    ratio = params.get('ratio', 0.01)
    if model_params == 'none':
        sys.exit("No model parameters for grace_from_params()")
    fusion_num = params.get('fusion_num', 2)
    momentum = params.get('momentum', 0.9)
    qsgd_quantum = params.get('quantum', 64)

    print("[Compression Setup] compressor: {}\n\tmemory: {}\n\tcommunicator: {}\n\tsparsity ratio: {}\n\tfusion num: {}".format(
        comp,
        mem,
        comm,
        ratio,
        fusion_num
    ))

    if comp == 'dgc':
        from .compressor.pooldgc import PoolDgcCompressor
        compressor = PoolDgcCompressor(compress_ratio=ratio)
    elif comp == 'efsignsgd':
        from .compressor.poolefsignsgd import PoolEFSignSGDCompressor
        compressor = PoolEFSignSGDCompressor()
    elif comp == 'fp16':
        from .compressor.poolfp16 import PoolFP16Compressor
        compressor = PoolFP16Compressor()
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

    if fusion_num > 0:
        if mem == 'dgc':
            from .memory.pooldgc import PoolDgcMemory
            if comp == 'topk' or comp == 'randomk':
                memory = PoolDgcMemory(model_params, momentum=0, fusion_num=fusion_num, momentum_masking=False)
            else:
                memory = PoolDgcMemory(model_params, fusion_num=fusion_num)
        elif mem == 'none':
            from .memory.poolnone import PoolNoneMemory
            memory = PoolNoneMemory(model_params, fusion_num=fusion_num)
        elif mem == 'residual':
            from .memory.poolresidual import PoolResidualMemory
            memory = PoolResidualMemory(model_params, fusion_num=fusion_num)
        else:
            raise NotImplementedError(mem)
    elif fusion_num == 0:
        if mem == 'dgc':
            from .memory.dgc import DgcMemory
            if comp == 'topk' or comp == 'randomk':
                memory = DgcMemory(model_params, momentum=0, momentum_masking=False)
            else:
                memory = DgcMemory(model_params)
        elif mem == 'none':
            from .memory.none import NoneMemory
            memory = NoneMemory(model_params)
        elif mem == 'residual':
            from .memory.residual import ResidualMemory
            memory = ResidualMemory(model_params)
        else:
            raise NotImplementedError(mem)

    if comm == 'allreduce':
        from .communicator.pool_allreduce import PoolAllreduce
        return PoolAllreduce(compressor, memory)
    elif comm == 'allgather':
        from .communicator.pool_allgather import PoolAllgather
        return PoolAllgather(compressor, memory)
    elif comm == 'byteps':
        from .communicator.pool_byteps import PoolBytePS
        return PoolBytePS(compressor, memory)
    else:
        raise NotImplementedError(comm)



def add_parser_arguments(parser):
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
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
    parser.add_argument('--comm', type=str, default='allgather',
                        help='communication for compression')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use Adam optimizer')
    parser.add_argument('--speed-test', action='store_true', default=False,
                        help='test the training speed')

    parser.add_argument('--scheduler', action='store_true', default=False,
                        help='use scheduler partition')
    parser.add_argument('--scheduler-epoch', type=int, default=20,
                        help='scheduler AVERAGE for epoch')
    parser.add_argument('--scheduler-step', type=int, default=2,
                        help='scheduler add step for search')
    parser.add_argument('--scheduler-warmup', type=int, default=5,
                        help='warmup iterations for scheduler')
    parser.add_argument('--scheduler-baseline', action='store_true', default=False,
                        help='use scheduler baseline based on tensor number')
    return parser


def set_compressor(model, args):
    if args.compress:
        """
        compressor: dgc, efsignsgd, fp16, none, onebit, qsgd, randomk, signsgd, signum, terngrad, threshold, topk
        memory: dgc, none, residual, 1bitadam.   Note: 1bitadam is for Adam
        comm: allreduce, allgather, ps
        """
        params = {
            'compressor': args.compressor,
            'memory': args.memory,
            'communicator': args.comm,
            'params': model.trainable_variables,
            'ratio': args.compress_ratio,
            'fusion_num': args.fusion_num,
        }

        return byteps_from_params(params)

    else:
        # Horovod: (optional) compression algorithm.
        return bps.Compression.fp16 if args.fp16 else bps.Compression.none
