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
