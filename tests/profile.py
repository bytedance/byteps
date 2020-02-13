import mxnet as mx


import byteps.mxnet as bps

bps.init()

mx.random.seed(2020)
shapes = [2<<x for x in range(24)]

ctx = mx.gpu(0)
for i, shape in enumerate(shapes):
    w = mx.gluon.Parameter('test', shape=shape, init=mx.init.Xavier())
    w.initialize(ctx=ctx)
    setattr(w, "byteps_compressor_type", "onebit")
    byteps_params = dict(
        filter(lambda attr: attr[0].startswith(
            "byteps_",), param.__dict__.items())
    )
    bps.byteps_declare_tensor("gradient_" + str(i), **byteps_params)
    bps.byteps_push_pull(tensor, name="gradient_" + str(i))