import mxnet as mx
while True:
    x = mx.nd.ones((5000, 5000)).copyto(mx.gpu())
    y = mx.nd.dot(x, x)
    print(y)

