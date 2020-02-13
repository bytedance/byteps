import mxnet as mx


import byteps.mxnet as bps

bps.init()

mx.random.seed(2020)
shapes = [2<<x for x in range(18)]
extra = list(map(lambda x:3*x, shapes))
shapes.extend(extra)
shapes *= 10 # test 10 times
print(shapes)

ctx = mx.gpu(0)
for i, shape in enumerate(shapes):
    w = mx.gluon.Parameter('test', shape=shape, init=mx.init.Normal())
    w.initialize(ctx=ctx)
    setattr(w, "byteps_compressor_type", "onebit")
    byteps_params = dict(
        filter(lambda attr: attr[0].startswith(
            "byteps_",), w.__dict__.items())
    )
    p = w._check_and_get(w._data, list)
    bps.byteps_declare_tensor("gradient_" + str(i), **byteps_params)
    bps.byteps_push_pull(p[0], name="gradient_" + str(i))
    p[0].wait_to_read()