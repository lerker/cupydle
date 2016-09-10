from cupydle.dnn.utils_theano import gpu_info
print(gpu_info(conversion='Mb'))

import theano
import numpy

# dropout
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from numpy.random import randint as npRandint
theanoGenerator = RandomStreams(seed=npRandint(1, 1000))

visibleDropout = 1.0
tam = (1,10)
dropoutMaskVisible = theanoGenerator.binomial(size=tam,
                                                n=1, p=visibleDropout,
                                                dtype=theano.config.floatX)


coso = theano.function([],dropoutMaskVisible)
for i in range(10):
    print(coso())


visibles = 5
ocultas = 4

matrix = numpy.asarray(numpy.random.randn(visibles,ocultas), dtype=numpy.float32)
vector = numpy.asarray(numpy.random.randn(ocultas), dtype=numpy.float32)
entrada = numpy.asarray(numpy.random.randn(visibles), dtype=numpy.float32)

print("entrda\n",entrada)
print()
print("matrix\n",matrix)
print()
print("vector\n",vector)
print()

# x*W+b
# (1,5)*(5,4)+(1,4)
lineal = numpy.dot(entrada, matrix) + vector
print("lineal\n",lineal)

# (1,5)
mask = numpy.asarray(numpy.asarray([1,0,1,1,0]), dtype=numpy.float32)
print("mask\n",mask)

# (1,5)*(1,5)
entradaMask = mask*entrada
print("mask*entrada\n", entradaMask)


v = theano.tensor.fvector()
b = theano.tensor.fvector()
m = theano.tensor.fvector()
w = theano.tensor.fmatrix()
w2 = theano.tensor.fmatrix()
w3 = theano.tensor.fmatrix()
s = theano.dot(v, w) + b

t = m * v # entrada enmascarada

mask2 = numpy.asarray(numpy.asarray([1,0,1,1,1]), dtype=numpy.float32)

masMatrix =w.T*m

from cupydle.dnn.loss import errorCuadraticoMedio as mse
v1 = theano.tensor.fvector()
v2 = theano.tensor.fvector()

mse = theano.tensor.mean(theano.tensor.sum((w2 - w3) ** 2))

coso = theano.function([v,w,b,m, w2, w3],[s,t, masMatrix, mse])

print("\n\n")
ww2 = numpy.asarray([[1.0, 1.5],[4.0, 3.2]], dtype=numpy.float32)
ww3 = numpy.asarray([[1.0, 0.5],[1.0, 1.0]], dtype=numpy.float32)
print("\n", ww2, "\n", ww3)

salida1, salida2 , salida3, salida4= coso(entrada,matrix, vector, mask, ww2, ww3)

print(salida1)
print()
print(salida2)
print()
print(salida3)
print()
print(salida4)


print(((ww2-ww3)**2))



print("\n\n\n")

#ll = theanoGenerator.normal(size=(1,4), avg=0.0, std=1.0)
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
rv_u = srng.uniform((1,4))
rv_n = srng.normal((1,4))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
#nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

#print(nearly_zeros())
print(g())
#coso = theano.function([],ll)
for i in range(10):
    print(f())


from cupydle.dnn.unidades import UnidadGaussiana_prueba as ug

unit = ug(media=0.0, desviacionEstandar=0.1)
unit.dibujar_histograma()

from cupydle.dnn.funciones import gaussianaTheano as gT
prob = gT(media=0.0, desviacionEstandar=1.0)
prob.dibujar()

for i in range(20):
    cooo = numpy.random.normal(size=(1,5), loc=0.0, scale=0.1)
    print(cooo)

mu, sigma = 0, 0.1 # mean and standard deviation
s = numpy.random.normal(mu, sigma, 1000)
print(abs(mu - numpy.mean(s)) < 0.01)

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()




###
###
"""
import numpy as np
import theano.tensor as T
T.config.floatX = 'float32'
dataPoints = np.random.random((50000, 784)).astype(T.config.floatX)
#float32 data type requires 4 bytes
sizeinGBs = 5000 * 784 * 4 / 1024. / 1024 / 1024 + 0
print("Data will need GBs of free memory", sizeinGBs)
"""


from cupydle.dnn.utils_theano import gpu_info

import numpy as np
import theano.tensor as T
from theano import shared
T.config.floatX = 'float32'

import time

print("inicio")
print(gpu_info(conversion='Mb'))
testData = shared(np.random.random((50000, 1000)).astype(T.config.floatX), borrow = True)
print("despues de 1", gpu_info(conversion='Mb'))
time.sleep(10)
del testData
print("despues de 2", gpu_info(conversion='Mb'))
time.sleep(10)
testData = shared(np.random.random((50000, 1000)).astype(T.config.floatX), borrow = False)
print("despues de 3", gpu_info(conversion='Mb'))
time.sleep(10)















If you're not revisiting datapoints then there probably isn't any value in using shared variables.

The following code could be modified and used to evaluate the different methods of getting data into your specific computation.

The "input" method is the one that will probably be best when you have no need to revisit data. The "shared_all" method may outperform everything else but only if you can fit the entire dataset in GPU memory. The "shared_batched" allows you to evaluate whether hierarchically batching your data could help.

In the "shared_batched" method, the dataset is divided into many macro batches and each macro batch is divided into many micro batches. A single shared variable is used to hold a single macro batch. The code evaluates all the micro batches within the current macro batch. Once a complete macro batch has been processed the next macro batch is loaded into the shared variable and the code iterates over the micro batches within it again.

In general, it might be expected that small numbers of large memory transfers will operate faster than larger numbers of smaller transfers (where the total transfered is the same for each). But this needs to be tested (e.g. with the code below) before it can be known for sure; YMMV.

The use of the "borrow" parameter may also have a significant impact on the performance, but be aware of the implications before using it.





import math
import timeit
import numpy
import theano
import theano.tensor as tt


def test_input(data, batch_size):
    assert data.shape[0] % batch_size == 0
    batch_count = data.shape[0] / batch_size
    x = tt.tensor4()
    f = theano.function([x], outputs=x.sum())
    total = 0.
    start = timeit.default_timer()
    for batch_index in xrange(batch_count):
        total += f(data[batch_index * batch_size: (batch_index + 1) * batch_size])
    print 'IN\tNA\t%s\t%s\t%s\t%s' % (batch_size, batch_size, timeit.default_timer() - start, total)


def test_shared_all(data, batch_size):
    batch_count = data.shape[0] / batch_size
    for borrow in (True, False):
        start = timeit.default_timer()
        all = theano.shared(data, borrow=borrow)
        load_time = timeit.default_timer() - start
        x = tt.tensor4()
        i = tt.lscalar()
        f = theano.function([i], outputs=x.sum(), givens={x: all[i * batch_size:(i + 1) * batch_size]})
        total = 0.
        start = timeit.default_timer()
        for batch_index in xrange(batch_count):
            total += f(batch_index)
        print 'SA\t%s\t%s\t%s\t%s\t%s' % (
            borrow, batch_size, batch_size, load_time + timeit.default_timer() - start, total)


def test_shared_batched(data, macro_batch_size, micro_batch_size):
    assert data.shape[0] % macro_batch_size == 0
    assert macro_batch_size % micro_batch_size == 0
    macro_batch_count = data.shape[0] / macro_batch_size
    micro_batch_count = macro_batch_size / micro_batch_size
    macro_batch = theano.shared(numpy.empty((macro_batch_size,) + data.shape[1:], dtype=theano.config.floatX),
                                borrow=True)
    x = tt.tensor4()
    i = tt.lscalar()
    f = theano.function([i], outputs=x.sum(), givens={x: macro_batch[i * micro_batch_size:(i + 1) * micro_batch_size]})
    for borrow in (True, False):
        total = 0.
        start = timeit.default_timer()
        for macro_batch_index in xrange(macro_batch_count):
            macro_batch.set_value(
                data[macro_batch_index * macro_batch_size: (macro_batch_index + 1) * macro_batch_size], borrow=borrow)
            for micro_batch_index in xrange(micro_batch_count):
                total += f(micro_batch_index)
        print 'SB\t%s\t%s\t%s\t%s\t%s' % (
            borrow, macro_batch_size, micro_batch_size, timeit.default_timer() - start, total)


def main():
    numpy.random.seed(1)

    shape = (20000, 3, 32, 32)

    print 'Creating random data with shape', shape
    data = numpy.random.standard_normal(size=shape).astype(theano.config.floatX)

    print 'Running tests'
    for macro_batch_size in (shape[0] / pow(10, i) for i in xrange(int(math.log(shape[0], 10)))):
        test_shared_all(data, macro_batch_size)
        test_input(data, macro_batch_size)
        for micro_batch_size in (macro_batch_size / pow(10, i) for i in
                                 xrange(int(math.log(macro_batch_size, 10)) + 1)):
            test_shared_batched(data, macro_batch_size, micro_batch_size)


main()
