
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
import theano.tensor.nnet
from theano.tensor import erf
from theano.tensor import sqrt
from theano.tensor import exp
from numpy import tanh
from numpy import exp
import numpy.random

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

#TODO
"THEANO..."


#TODO ver esta implementacion de la de abajo
##from theano.tensor.shared_randomstreams import RandomStreams

#from theano.tensor.shared_randomstreams import RandomStreams  #random seed CPU
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams # GPU

## ultra rapido
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU - GPU
                                                                        #(parece que binomial no esta implementado, lo reemplaza por uniform)
                                                                        # cambiar a: multinomial(size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None)[source]
                                                                        # en activationFunction

theanoFloat  = theano.config.floatX

class ActivationFunction(object):
    # TODO ver esto de generacion de randoms cada vez

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        #if 'theanoGenerator' in odict:
        #    del odict['theanoGenerator']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)   # update attributes

    def __getinitargs__():
        return None


class Sigmoid(ActivationFunction):
    def __init__(self):
        #self.theanoGenerator = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=numpy.random.randint(1, 1000))
        # o la version  MRG31k3p
        self.theanoGenerator = RandomStreams(seed=numpy.random.randint(1, 1000))

    def deterministic(self, x):
        return theano.tensor.nnet.sigmoid(x)

    def nonDeterminstic(self, x):
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
        val = self.deterministic(x)
        return self.theanoGenerator.binomial(size=val.shape, n=1, p=val, dtype=theanoFloat)

    def activationProbablity(self, x):
        return self.deterministic(x)

class Rectified(ActivationFunction):

  def __init__(self):
    pass

  def nonDeterminstic(self, x):
    return self.deterministic(x)

  def deterministic(self, x):
    return x * (x > 0.0)

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
