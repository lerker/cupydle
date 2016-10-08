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
from numpy.random import uniform as npUniform

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU - GPU
                                                                        #(parece que binomial no esta implementado, lo reemplaza por uniform)
                                                                        # cambiar a: multinomial(size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None)[source]
                                                                        # en activationFunction

theanoFloat  = Tconfig.floatX

from cupydle.dnn.funciones import sigmoideaTheano
from cupydle.dnn.funciones import gaussianaTheano

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
        return ("Unidad Binaria")


class UnidadBinariaDropout(Unidad):
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

    def activar(self, x, p):
        probability = self.deterministico(x)
        mask = self.theanoGenerator.binomial(probability.shape, n=1, p=p)
        activacion = self.theanoGenerator.binomial(size=probability.shape, n=1, p=probability, dtype=theanoFloat)
        return mask*activacion, probability

    def __str__(self):
        return ("Unidad Binaria")

class UnidadGaussiana(Unidad):
    def __init__(self, media=0.0, desviacionEstandar=1.0, factor=1.0):
        # inicializar la clase padre
        super(UnidadGaussiana, self).__init__()
        #self.theanoGenerator = RandomStreams(seed=npRandint(1, 1000))
        # aca los metodos propios de esta clase
        self.fn = gaussianaTheano(media=media, desviacionEstandar=desviacionEstandar, factor=factor)

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
        return self.theanoGenerator.normal(size=probability.shape, n=1, p=probability, dtype=theanoFloat), probability

    def probabilidadActivacion(self, x):
        return self.deterministico(x)

    def activar(self, x):
        """
        normal(self, size=(), avg=0.0, std=1.0, ndim=None):
        Sample from a normal distribution centered on avg with the specified standard deviation (std)

        If size is ambiguous on the number of dimensions, ndim may be a plain integer to supplement the missing information.

        This wrap numpy implementation, so it have the same behavior.
        """
        probability = self.deterministico(x)
        return self.theanoGenerator.normal(size=probability.shape, avg=self.fn.media, std=self.fn.desviacionEstandar), probability

    def dibujar(self, axe=None, axis=[-10.0, 10.0],
                axline=[0.0, 0.0], mostrar=True):
        import matplotlib.pylab as plt
        from theano.tensor import dvector as Tdvector
        from theano import function as Tfunction

        if axe is None:
          axe = plt.gca()

        Xaxis = npUniform(low=0.0, high=1.0, size=1000)
        x = Tdvector('x')
        Yaxis, prob = self.activar(x)

        dibujador=Tfunction(inputs=[x], outputs=Yaxis)
        axe.plot(Xaxis, dibujador(Xaxis), color='red', marker='.', linestyle='None')
        #axe.plot(Xaxis, prob, color='blue', marker='.', linestyle='None')
        # lineas horizontales y verticales
        axe.axhline(axline[0], linestyle='-.', color='blue', linewidth=1.5)
        axe.axvline(axline[1], linestyle='-.', color='blue', linewidth=1.5)
        plt.grid(True)

        plt.show() if mostrar else None
        return 1

    def dibujar_histograma(self, axe=None, mostrar=True):
        import matplotlib.pyplot as plt
        from theano.tensor import dvector as Tdvector
        from theano import function as Tfunction
        import numpy

        if axe is None:
            axe = plt.gca()

        Xaxis = npUniform(low=0.0, high=1.0, size=1000)
        x = Tdvector('x')
        s, prob = self.activar(x)
        dibujador=Tfunction(inputs=[x], outputs=s)
        s=dibujador(Xaxis)
        mu = self.fn.media
        sigma = self.fn.desviacionEstandar

        #s = numpy.random.normal(mu, sigma, 1000)
        print("chequeo <media - media_conjunto>: ", abs(mu - numpy.mean(s)) < 0.01)

        count, bins, ignored = plt.hist(s, 30, normed=True)
        plt.plot(bins, 1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

        plt.show() if mostrar else None
        return 1

    def __str__(self):
        return ("Unidad Gaussiana")




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

class UnidadGaussiana_prueba(Unidad):
    def __init__(self, media=0.0, desviacionEstandar=1.0, factor=1.0):
        # inicializar la clase padre
        super(UnidadGaussiana_prueba, self).__init__()
        #self.theanoGenerator = RandomStreams(seed=npRandint(1, 1000))
        # aca los metodos propios de esta clase
        self.fn = gaussianaTheano(media=media, desviacionEstandar=desviacionEstandar, factor=factor)

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
        return self.theanoGenerator.normal(size=probability.shape, n=1, p=probability, dtype=theanoFloat), probability

    def probabilidadActivacion(self, x):
        return self.deterministico(x)

    def activar(self, x):
        """
        normal(self, size=(), avg=0.0, std=1.0, ndim=None):
        Sample from a normal distribution centered on avg with the specified standard deviation (std)

        If size is ambiguous on the number of dimensions, ndim may be a plain integer to supplement the missing information.

        This wrap numpy implementation, so it have the same behavior.
        """
        probability = self.deterministico(x)
        retorno = self.theanoGenerator.normal(size=x.shape, avg=self.fn.media, std=self.fn.desviacionEstandar)
        return retorno, retorno

    def dibujar(self, axe=None, axis=[-10.0, 10.0],
                axline=[0.0, 0.0], mostrar=True):
        import matplotlib.pylab as plt
        from theano.tensor import dvector as Tdvector
        from theano import function as Tfunction

        if axe is None:
          axe = plt.gca()

        Xaxis = npUniform(low=0.0, high=1.0, size=1000)
        x = Tdvector('x')
        Yaxis, prob = self.activar(x)

        dibujador=Tfunction(inputs=[x], outputs=Yaxis)
        axe.plot(Xaxis, dibujador(Xaxis), color='red', marker='.', linestyle='None')
        #axe.plot(Xaxis, prob, color='blue', marker='.', linestyle='None')
        # lineas horizontales y verticales
        axe.axhline(axline[0], linestyle='-.', color='blue', linewidth=1.5)
        axe.axvline(axline[1], linestyle='-.', color='blue', linewidth=1.5)
        plt.grid(True)

        plt.show() if mostrar else None
        return 1

    def dibujar_histograma(self, axe=None, axis=[-10.0, 10.0],
                axline=[0.0, 0.0], mostrar=True):
        import matplotlib.pyplot as plt
        from theano.tensor import dvector as Tdvector
        from theano import function as Tfunction
        import numpy

        if axe is None:
            axe = plt.gca()

        Xaxis = npUniform(low=0.0, high=1.0, size=1000)
        x = Tdvector('x')
        s, prob = self.activar(x)
        dibujador=Tfunction(inputs=[x], outputs=s)
        s=dibujador(Xaxis)
        mu = self.fn.media
        sigma = self.fn.desviacionEstandar

        #mu, sigma = 0, 0.1 # mean and standard deviation
        #s = numpy.random.normal(mu, sigma, 1000)
        print("chequeo media - media_conjunto: ", abs(mu - numpy.mean(s)) < 0.01)

#        import matplotlib.pyplot as plt
        count, bins, ignored = plt.hist(s, 30, normed=True)
        plt.plot(bins, 1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

        plt.show() if mostrar else None
        return 1

    def __str__(self):
        return ("Unidad Gaussiana EXPERIMENTAL")
