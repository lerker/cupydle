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

"""
import numpy

import theano

from cupydle.dnn.activations import Sigmoid


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
            del W_values
        else:
            if type(W).__module__ != numpy.__name__:
                assert False, "Solo acepto del tipo numpy.ndarray"
            else:
                W = theano.shared(value=W, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((nOut,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            del b_values
        else:
            if type(b).__module__ != numpy.__name__:
                assert False, "Solo acepto del tipo numpy.ndarray"
            else:
                b = theano.shared(value=b, name='b', borrow=True)

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

    # propiedades intrisecas de las capas
    def __str__(self):
        return str("Capa: " + str(type(self))
                    + "\n   W[" + str(self.W) + "]:   "
                    + str(self.W.get_value(borrow=True).shape)
                    + "   " + str(type(self.W))
                    + "\n   bias[" + str(self.b) + "]:"
                    + str(type(self.b.get_value(borrow=True).shape))
                    + "   " + str(type(self.b)))

    # funciones para obtener valores
    def get_weights(self):
        return self.W

    def get_bias(self):
        return self.b

    def set_weights(self, w):
        if isinstance(w, theano.TensorType):
            self.W.set_value(w)
        else:
            assert False

    def set_bias(self, b):
        if isinstance(b, theano.TensorType):
            self.b.set_value(b)
        else:
            assert False


class ClassificationLayer(Layer):
    def __init__(self, nIn, nOut, input, W=None, b=None):
        # initialize with 0 the weights W as a matrix of shape (nIn, nout)

        if W is None:
            W_values = numpy.zeros((nIn, nOut), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
            del W_values
        else:
            if type(W).__module__ != numpy.__name__:
                assert False, "Solo acepto del tipo numpy.ndarray"
            else:
                W = theano.shared(value=W, name='W', borrow=True)

        # initialize the biases b as a vector of nOut 0s
        if b is None:
            b_values = numpy.zeros((nOut,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            del b_values
        else:
            if type(b).__module__ != numpy.__name__:
                assert False, "Solo acepto del tipo numpy.ndarray"
            else:
                b = theano.shared(value=b, name='b', borrow=True)

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

if __name__ == '__main__':
    assert False, "Este modulo no es ejecutable!!!"