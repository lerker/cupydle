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
unidades para las rbm
"""
from abc import ABC, abstractmethod

from theano import config as Tconfig
import theano.tensor.nnet as Tnet
from theano.tensor import erf as Terf
from theano.tensor import sqrt as Tsqrt
from theano.tensor import exp as Texp
from numpy import tanh as npTanh
from numpy import exp as npExp
from numpy.random import RandomState as npRandom
from numpy.random import randint as npRandint

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU - GPU
                                                                        #(parece que binomial no esta implementado, lo reemplaza por uniform)
                                                                        # cambiar a: multinomial(size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None)[source]
                                                                        # en activationFunction

theanoFloat  = Tconfig.floatX

from cupydle.dnn.funciones import sigmoideaTheano

# clase abstracta
class Unidad(ABC):
    def __init__(self):
        #self.theanoGenerator = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=numpy.random.randint(1, 1000))
        # o la version  MRG31k3p
        self.theanoGenerator = RandomStreams(seed=npRandint(1, 1000))

    """
    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        #if 'theanoGenerator' in odict:
        #    del odict['theanoGenerator']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)   # update attributes

    def __getinitargs__():
        return None
    """

    @abstractmethod
    def activar(self, x):
        return x

    @abstractmethod
    def probabilidadActivacion(self):
        return 1

    @abstractmethod
    def __str__(self):
        return "Unidad"


class UnidadBinaria(Unidad):
    def __init__(self):
        # inicializar la clase padre
        super(UnidadBinaria, self).__init__()
        #self.theanoGenerator = RandomStreams(seed=npRandint(1, 1000))
        # aca los metodos propios de esta clase
        #self.__baz = 21
        self.fn = sigmoideaTheano()

    def deterministico(self, x):
        return self.fn(x)

    def noDeterministico(self, x):
        # TODO por lo visto la binomial no esta implementada en CUDA,
        # por lo tanto lo lleva a la GPU a los datos
        # luego calcula la binomial (la cual la trae a la CPU de nuevo los datos)
        # y por ultimo lleva de nuevo a la GPU los datos calculado
        ### SOLUCION
        # deberia calcularse los numeros binomiales ({0,1}) en la GPU sin usar RandomStreams.binomial
        # si se retorna val y al theano.tensor.nnet.sigmoid(x) se le agrega 'transfer('gpu')' de la
        # activationProbability en el grafo se da cuenta de la optimizacion
        ###$
        # http://deeplearning.net/software/theano/tutorial/examples.html#example-other-random
        # There are 2 other implementations based on MRG31k3p and CURAND.
        # The RandomStream only work on the CPU, MRG31k3p work on the CPU and GPU. CURAND only work on the GPU.
        probability = self.deterministico(x)
        return self.theanoGenerator.binomial(size=probability.shape, n=1, p=probability, dtype=theanoFloat), probability

    def probabilidadActivacion(self, x):
        return self.deterministico(x)

    def activar(self, x):
        probability = self.deterministico(x)
        return self.theanoGenerator.binomial(size=probability.shape, n=1, p=probability, dtype=theanoFloat), probability

    def __str__(self):
        return ("Unidad Binaria Sigmoidea")

"""
# ver esta todo medio mal
class RectificadorRuidoso(Unidad):
    def __init__(self):
        # inicializar la clase padre
        super(RectificadorRuidoso, self).__init__()
        #self.theanoGenerator = RandomStreams(seed=npRandint(1, 1000))
        # aca los metodos propios de esta clase
        #self.__baz = 21
        self.fn = sigmoideaTheano()

    def deterministico(self, x):
        return self.fn(x)

    def noDeterministico(self, x):
        probability = self.deterministico(x)
        return self.theanoGenerator.binomial(size=probability.shape, n=1, p=probability, dtype=theanoFloat), probability

    def probabilidadActivacion(self, x):
        return 1.0 - self.cdf(0, miu=x, variance=theano.tensor.nnet.sigmoid(x))

    def activar(self, x):
        x += self.theanoGenerator.normal(avg=0.0, std=(theano.tensor.sqrt(theano.tensor.nnet.sigmoid(x)) + 1e-8))
        return x * (x > 0.0)


    def expectedValueRectified(self, mean, variance):
        std = Tsqrt(variance)
        return std / Tsqrt(2.0 * np.pi) * Texp(- mean**2 / (2.0 * variance)) + mean * self.cdf(mean / std)

    def __str__(self):
        return "Unidad Rectificador Ruidoso"


    def nonDeterminstic(self, x):
        x += self.theanoGenerator.normal(avg=0.0, std=(theano.tensor.sqrt(theano.tensor.nnet.sigmoid(x)) + 1e-8))
        return x * (x > 0.0)

    def deterministic(self, x):
        return expectedValueRectified(x, theano.tensor.nnet.sigmoid(x) + 1e-08)

    # Approximation of the cdf of a standard normal
    def cdf(self, x, miu=0.0, variance=1.0):
        return 1.0/2 *  (1.0 + Terf((x - miu)/ Tsqrt(2 * variance)))
"""
