import numpy as np

import cupydle.dnn.activations as act
from cupydle.dnn.Neurons import Neurons as Neurons

__author__ = "Nelson Ponzoni"
__copyright__ = "Copyright 2015-2016, Proyecto Final de Carrera"
__credits__ = ["Nelson Ponzoni"]
__license__ = "GPL"
__version__ = "20160101"
__maintainer__ = "Nelson Ponzoni"
__email__ = "npcuadra@gmail.com"
__status__ = "Production"

"""

"""
import theano

from cupydle.dnn.activations import Sigmoid
import numpy

class Layer(object):
    def __init__(self, nIn, nOut, input, rng, activationFn, W=None, b=None):
        self.activationFn = activationFn
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (nIn + nOut)),
                    high=numpy.sqrt(6. / (nIn + nOut)),
                    size=(nIn, nOut)
                ),
                dtype=theano.config.floatX
            )
            if type(self.activationFn) == type(Sigmoid()):
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((nOut,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.activationFn=activationFn

        # parameters of the model
        self.params = [self.W, self.b]

        self.x = input

    def activate(self):
        lin_output = theano.tensor.dot(self.x, self.W) + self.b
        #output = (lin_output if self.activationFn is None else self.activationFn.deterministic(lin_output))
        output = self.activationFn.deterministic(lin_output)
        return output

class classificationLayer(Layer):
    def __init__(self, nIn, nOut, input, W=None, b=None):
        # initialize with 0 the weights W as a matrix of shape (nIn, nout)

        if W is None:
            W_values = numpy.zeros((nIn, nOut), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
            del W_values

        # initialize the biases b as a vector of nOut 0s
        if b is None:
            b_values = numpy.zeros((nOut,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            del b_values

        self.W = W
        self.b = b

        # parameters of the model
        self.params = [self.W, self.b]

        self.x = input

    def activate(self):
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        return theano.tensor.nnet.softmax(theano.tensor.dot(self.x, self.W) + self.b)

    def predict(self):
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        return theano.tensor.argmax(self.activate(), axis=1)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -theano.tensor.mean(theano.tensor.log(self.activate())[theano.tensor.arange(y.shape[0]), y])


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.predict().ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.predict().type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return theano.tensor.mean(theano.tensor.neq(self.predict(), y))
        else:
            raise NotImplementedError()



class NeuralLayer(object):
    def __init__(self,
                 n_in=2,
                 n_out=2,
                 activation='Sigmoid',
                 w=None,
                 b=None,
                 rng=None):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = act.activation_functions[activation]  # activacion de la capa
        self.activation_d = act.activation_functions_prime[activation]  # derivada de la activacion de la capa

        # no le di semilla?
        if rng is None:
            rng = np.random.seed(1234)

        self.rng = rng
        self.shape_w = n_out, n_in  # tupla
        self.shape_b = n_out, 1

        # si los pesos no fueron pasados, es porque hay que inicializarlos
        if w is None:
            # inicializacion media rara de deeplearning.com
            w = np.asarray(
                np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                                  high=+np.sqrt(6.0 / (n_in + n_out)),
                                  size=self.shape_w),
                dtype=np.dtype(float))
        if b is None:
            b = np.zeros(self.shape_b, dtype=np.dtype(float))

        self.weights = Neurons(w, self.shape_w)
        self.bias = Neurons(b, self.shape_b)

    # propiedades intrisecas de las capas
    def __str__(self):
        return str("W[" + str(self.weights.shape) + "]:\n" + str(self.weights) + "\nbias[" + str(self.bias.shape) + "]:\n" + str(self.bias))

    def __mul__(self, other):
        self.weights *= other
        self.bias *= other
        return self

    # funciones para obtener valores
    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def set_weights(self, w):
        if isinstance(w, Neurons):
            self.weights = w
        else:
            self.weights = Neurons(w, self.shape_w)

    def set_bias(self, b):
        if isinstance(b, Neurons):
            self.bias = b
        else:
            self.bias = Neurons(b, self.shape_b)

    def output(self, x, grad=False):
        """
        Activacion de una capa con una entrada, si pido el gradiente lo retorno como una tupla
        :param x: entrada a la capa, debe ser una neurona
        :param grad:
        :return:
        """
        # multiplicacion de los pesos por la entrada
        wx = self.weights.mul_array(x)
        # adicion del bias
        z = wx.sum_array(self.bias)
        # a la salida lineal, es pasada por la funcion de activacion de transferencia
        a = z.activation(self.activation)
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)

        return a


    def update(self, step_w, step_b):
        # Actualiza sumando los argumentos w y b a los respectivos pesos
        self.weights += step_w
        # self.weights_T += step_w.transpose()
        self.bias += step_b
        return

    def outputLinear(self, x):
        """
        Activacion de una capa con una entrada, si pido el gradiente lo retorno como una tupla
        :param x: entrada a la capa, debe ser una neurona
        :param grad:
        :return:
        """
        # multiplicacion de los pesos por la entrada
        wx = self.weights.mul_array(x)
        # adicion del bias
        z = wx.sum_array(self.bias)
        # a la salida lineal, es pasada por la funcion de activacion de transferencia
        return z



class ClassificationLayer(NeuralLayer):
    def output(self, x, grad=False):
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.softmax()  # La activacion es un clasificador softmax
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
        return a
