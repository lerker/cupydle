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
funciones de activacion, para las nueronas/unidades
"""

#import theano
#import numpy
from theano.tensor import nnet as Tnet
from theano.tensor import erf as Terf
from theano.tensor import sqrt as Tsqrt
from theano.tensor import exp as Texp
from theano.tensor import tanh as Ttanh
from theano.tensor import dvector as Tdvector
from theano.tensor import cast as Tcast
from theano import function as Tfunction
from theano import config as Tconfig

import numpy.random as npRandom
from numpy import arange as npArange
from numpy import linspace as npLinspace
from numpy import tanh as npTanh
from numpy import exp as npExp

from abc import ABC, abstractmethod

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
        raise NotImplementedError()

    @abstractmethod
    def dibujar(self):
        raise NotImplementedError()

class identidadTheano(Funcion):
    def __call__(self, x):
        return x

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        x = Tdvector('x')
        s = Tcast(self(x), dtype=theanoFloat)
        dibujador=Tfunction(inputs=[x], outputs=s)
        plt.plot(Xaxis, dibujador(Xaxis), color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()

    def __str__(self):
        return "Identidad Theano"


class sigmoideaTheano(Funcion):
    def __call__(self, x):
        return Tnet.sigmoid(x)

    def dibujar(self):
        #super(sigmoideaTheano, self).dibujar()
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        x = Tdvector('x')
        s = Tcast(self(x), dtype=theanoFloat)
        dibujador=Tfunction(inputs=[x], outputs=s)
        plt.plot(Xaxis, dibujador(Xaxis), color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0.5, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Sigmoidea Theano"

class linealRectificadaTheano(Funcion):
    def __call__(self, x):
        #return x * (x > 0.0)
        return Tnet.relu(x)

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        x = Tdvector('x')
        s = Tcast(self(x), dtype=theanoFloat)
        dibujador=Tfunction(inputs=[x], outputs=s)
        plt.plot(Xaxis, dibujador(Xaxis), color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()

    def __str__(self):
        return "Lineal Rectificada Theano"

class tanhTheano(Funcion):
    def __call__(self, x):
        return Ttanh(x)

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        x = Tdvector('x')
        s = Tcast(self(x), dtype=theanoFloat)
        dibujador=Tfunction(inputs=[x], outputs=s)
        plt.plot(Xaxis, dibujador(Xaxis), color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Hiperbolica Theano"

class identidadNumpy(Funcion):
    def __call__(self, x):
        return x

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        plt.plot(Xaxis, Yaxis, color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Identidad Theano"


class sigmoideaNumpy(Funcion):
    def __call__(self, x):
        return 1.0 / (1.0 + npExp(-x))

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        plt.plot(Xaxis, Yaxis, color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0.5, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Sigmoidea Numpy"

class linealRectificadaNumpy(Funcion):
    def __call__(self, x):
        return x * (x > 0.0)

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        x = Tdvector('x')
        s = Tcast(self(x), dtype=theanoFloat)
        dibujador=Tfunction(inputs=[x], outputs=s)
        plt.plot(Xaxis, dibujador(Xaxis), color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()

    def __str__(self):
        return "Lineal Rectificada Numpy"

class tanhNumpy(Funcion):
    def __call__(self, x):
        return npTanh(x)

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        plt.plot(Xaxis,Yaxis, color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Hiperbolica Numpy"

class tanhDerivadaNumpy(Funcion):
    def __call__(self, x):
        return 1.0 - npTanh(x) ** 2

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        plt.plot(Xaxis,Yaxis, color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Derivada Hiperbolica Numpy"

class sigmoideaDerivadaNumpy(Funcion):
    def __call__(self, x):
        fn=sigmoideaNumpy()
        return fn(x) * (1.0 - fn(x))

    def dibujar(self):
        Xaxis = npArange(-10., 10., 0.01)
        Yaxis = self(Xaxis)
        plt.plot(Xaxis,Yaxis, color='red', linewidth=2.0)
        # lineas horizontales y verticales
        plt.axhline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.axvline(0, linestyle='-.', color='blue', linewidth=1.5)
        plt.title(self.__str__())
        plt.grid(True)
        plt.show()
        return 1

    def __str__(self):
        return "Derivada Sigmoidea Numpy"


# dictionaries to do the job more easy...
#activation_functions = {'Tanh': tanh, 'Sigmoid': sigmoid}
#activation_functions_prime = {'Tanh': tanh_prime, 'Sigmoid': sigmoid_prime}


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

