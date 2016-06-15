from theano import function, config, shared, sandbox
import theano.sandbox.cuda.basic_ops
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), 'float32'))
f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
print("Numpy result is %s" % (numpy.asarray(r),))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


"""
Returning a Handle to Device-Allocated Data
The speedup is not greater in the preceding example because the function is returning
its result as a NumPy ndarray which has already been copied from the device to the host
for your convenience. This is what makes it so easy to swap in device=gpu, but if you
don’t mind less portability, you might gain a bigger speedup by changing the graph to
express a computation with a GPU-stored result. The gpu_from_host op means “copy the
input from the host to the GPU” and it is optimized away after the T.exp(x) is
replaced by a GPU version of exp().


Here we’ve shaved off about 50% of the run-time by simply not copying the resulting array back to the host. The object returned by each function call is now not a NumPy array but a “CudaNdarray” which can be converted to a NumPy ndarray by the normal NumPy casting mechanism using something like numpy.asarray().

For even more speed you can play with the borrow flag. See Borrowing when Constructing Function Objects.
"""

