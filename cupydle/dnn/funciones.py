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
from theano.tensor import pow as Tpow
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

from cupydle.dnn.graficos import dibujarFnActivacionTheano
from cupydle.dnn.graficos import dibujarFnActivacionNumpy

theanoFloat  = Tconfig.floatX

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

#### --- funcion abstracta 'esqueleto' que todos deben implementar
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


##### --------------  THEANO



class identidadTheano(Funcion):
    #def __init__(self):
    #    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

    def __call__(self, x):
        return x

    def dibujar(self):
        dibujarFnActivacionTheano(self=self, axe=None, axis=[-10.0, 10.0],
                                  axline=[0.0, 0.0], mostrar=True)

    def __str__(self):
        return "Identidad Theano"


class sigmoideaTheano(Funcion):
    def __call__(self, x):
        return Tnet.sigmoid(x)

    def dibujar(self):
        #super(sigmoideaTheano, self).dibujar()
        dibujarFnActivacionTheano(self=self, axe=None, axis=[-10.0, 10.0],
                                  axline=[0.5, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Sigmoidea Theano"

class linealRectificadaTheano(Funcion):
    def __call__(self, x):
        #return x * (x > 0.0)
        return Tnet.relu(x)

    def dibujar(self):
        dibujarFnActivacionTheano(self=self, axe=None, axis=[-10.0, 10.0],
                                  axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Lineal Rectificada Theano"

class tanhTheano(Funcion):
    def __call__(self, x):
        return Ttanh(x)

    def dibujar(self):
        dibujarFnActivacionTheano(self=self, axe=None, axis=[-10.0, 10.0],
                                  axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Hiperbolica Theano"


class softmaxTheano(Funcion):
    # theano softmax es numericamente inestable? aparecen Nans... ver
    def __call__(self, x):
        e_x = Texp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def dibujar(self):
        raise NotImplementedError()
        #dibujarFnActivacionTheano(self=self, axe=None, axis=[-10.0, 10.0],
        #                          axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Softmax Theano"

class gaussianaTheano(Funcion):
    def __init__(self, media, desviacionEstandar, factor=1.0):
        #    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))
        # varianza = desviacionEstandar**2
        self.media=media
        self.desviacionEstandar=desviacionEstandar
        self.factor=factor

    def __call__(self, x):
        # funcion gaussiana se define como a*exp(- (x-mu)^2/(2*c^2) )
        exponente = Tpow((x-self.media),2)/(2*Tpow(self.desviacionEstandar,2))

        # funcion gaussiana normal..
        e_x = self.factor * Texp(-exponente)

        # si a es igual-> a= 1/sqrt(2*PI*c^2)
        # funcion de densidad de una variable aleatoria con distribucion normal de media mu=b y varianza std2=c2.
        #e_x = (1.0/(Tsqrt(2*3.141592*Tpow(self.desviacionEstandar,2)))) * Texp(-exponente)
        return e_x

    def dibujar(self):
        dibujarFnActivacionTheano(self=self, axe=None, axis=[-10.0, 10.0],
                                  axline=[0.0, 0.0], mostrar=True)

        return 1

    def __str__(self):
        return "Gaussiana Theano"



### --------- Numpy

class identidadNumpy(Funcion):
    def __call__(self, x):
        return x

    def dibujar(self):
        dibujarFnActivacionNumpy(self=self, axe=None, axis=[-10.0, 10.0],
                                 axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Identidad Numpy"


class sigmoideaNumpy(Funcion):
    def __call__(self, x):
        return 1.0 / (1.0 + npExp(-x))

    def dibujar(self):
        dibujarFnActivacionNumpy(self=self, axe=None, axis=[-10.0, 10.0],
                                 axline=[0.5, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Sigmoidea Numpy"

class linealRectificadaNumpy(Funcion):
    def __call__(self, x):
        return x * (x > 0.0)

    def dibujar(self):
        dibujarFnActivacionNumpy(self=self, axe=None, axis=[-10.0, 10.0],
                                 axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Lineal Rectificada Numpy"

class tanhNumpy(Funcion):
    def __call__(self, x):
        return npTanh(x)

    def dibujar(self):
        dibujarFnActivacionNumpy(self=self, axe=None, axis=[-10.0, 10.0],
                                 axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Hiperbolica Numpy"

class tanhDerivadaNumpy(Funcion):
    def __call__(self, x):
        return 1.0 - npTanh(x) ** 2

    def dibujar(self):
        dibujarFnActivacionNumpy(self=self, axe=None, axis=[-10.0, 10.0],
                                 axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Derivada Hiperbolica Numpy"

class sigmoideaDerivadaNumpy(Funcion):
    def __call__(self, x):
        fn=sigmoideaNumpy()
        return fn(x) * (1.0 - fn(x))

    def dibujar(self):
        dibujarFnActivacionNumpy(self=self, axe=None, axis=[-10.0, 10.0],
                                 axline=[0.0, 0.0], mostrar=True)
        return 1

    def __str__(self):
        return "Derivada Sigmoidea Numpy"


# dictionaries to do the job more easy...
#activation_functions = {'Tanh': tanh, 'Sigmoid': sigmoid}
#activation_functions_prime = {'Tanh': tanh_prime, 'Sigmoid': sigmoid_prime}


if __name__ == '__main__':
    assert False, "Este modulo no es ejecutable!!!"

