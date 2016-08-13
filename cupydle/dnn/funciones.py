#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"

"""
Activations functions, they will be used for 'activate' the signal of de neuron,
the linear signal is passed throw the activation function and loss the linearity.
"""

#import theano
#import numpy
import theano.tensor.nnet as Tnet
from theano.tensor import erf
from theano.tensor import sqrt
from theano.tensor import exp
from theano.tensor import dvector as Tdvector
from theano.tensor import cast as Tcast
from theano import function as Tfunction
from numpy import tanh
from numpy import exp
import numpy.random
from theano import config as Tconfig

from abc import ABC, abstractmethod
from theano import config as Tconfig

from numpy import arange as npArange
from numpy import linspace as npLinspace
import matplotlib.pylab as plt

theanoFloat  = Tconfig.floatX

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Funcion(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        return x

    @abstractmethod
    def __str__(self):
        return "Funcion"

    @abstractmethod
    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        x = Tdvector('x')
        s = Tcast(self(x), dtype=theanoFloat)
        dibujador=Tfunction(inputs=[x], outputs=s)
        plt.plot(Xaxis, dibujador(Xaxis))
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1


class sigmoideaTheano(Funcion):
    def __call__(self, x):
        return Tnet.sigmoid(x)

    def dibujar(self):
        super(sigmoideaTheano, self).dibujar()

    def __str__(self):
        return "Sigmoidea Theano"

class rectificadaTheano(Funcion):
    def __call__(self, x):
        return x * (x > 0.0)

    def dibujar(self):
        super(sigmoideaTheano, self).dibujar()



def tanh(z):
    return numpy.tanh(z)


def tanh_prime(z):
    return 1.0 - numpy.tanh(z) ** 2


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

# dictionaries to do the job more easy...
activation_functions = {'Tanh': tanh, 'Sigmoid': sigmoid}
activation_functions_prime = {'Tanh': tanh_prime, 'Sigmoid': sigmoid_prime}


"""
class RectifiedNoisy(ActivationFunction):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    x += self.theanoGenerator.normal(avg=0.0, std=(theano.tensor.sqrt(theano.tensor.nnet.sigmoid(x)) + 1e-8))
    return x * (x > 0.0)

  def deterministic(self, x):
    return expectedValueRectified(x, theano.tensor.nnet.sigmoid(x) + 1e-08)

  def activationProbablity(self, x):
    return 1.0 - cdf(0, miu=x, variance=theano.tensor.nnet.sigmoid(x))

class RectifiedNoisyVar1(ActivationFunction):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=numpy.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    x += self.theanoGenerator.normal(avg=0.0, std=1.0)
    return x * (x > 0.0)

  def deterministic(self, x):
    return expectedValueRectified(x, 1.0)

  def activationProbablity(self, x):
    return 1.0 - cdf(0, miu=x, variance=1.0)

class Identity(ActivationFunction):

  def deterministic(self, x):
    return x

class Softmax(ActivationFunction):

  def deterministic(self, v):
    # Do not use theano's softmax, it is numerically unstable
    # and it causes Nans to appear
    # Semantically this is the same
    e_x = theano.tensor.exp(v - v.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# TODO: try this for the non deterministic version as well
class CappedRectifiedNoisy(ActivationFunction):
    def __init__(self):
        pass

    def nonDeterminstic(self, x):
        return self.deterministic(x)

    def deterministic(self, x):
        return x * (x > 0.0) * (x < 6.0)

    # TODO
    def activationProbablity(self, x):
        return None

def expectedValueRectified(mean, variance):
    std = theano.tensor.sqrt(variance)
    return std / theano.tensor.sqrt(2.0 * numpy.pi) * theano.tensor.exp(- mean**2 / (2.0 * variance)) + mean * cdf(mean / std)

# Approximation of the cdf of a standard normal
def cdf(x, miu=0.0, variance=1.0):
    return 1.0/2 *  (1.0 + theano.tensor.erf((x - miu)/ theano.tensor.sqrt(2 * variance)))


"""
if __name__ == '__main__':
    assert False, "Este modulo no es ejecutable!!!"

