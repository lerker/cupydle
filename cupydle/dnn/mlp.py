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
Implementacion de un 'Multi-layer Perceptron' en GP-GPU/CPU (Theano)

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

#from __future__ import print_function
#__docformat__ = 'restructedtext en'


import os
import sys
import timeit
import pickle
import gzip

import numpy

import theano

from cupydle.dnn.activations import Sigmoid
from cupydle.dnn.activations import Rectified
from cupydle.dnn.layer import Layer
from cupydle.dnn.layer import ClassificationLayer
from cupydle.dnn.rbm_gpu import rbm_gpu
from cupydle.dnn.utils import save
from cupydle.dnn.utils import load as load_utils
from cupydle.dnn.utils_theano import shared_dataset



verbose = False
class MLP(object):

    def __init__(self, task, rng=None):

        # no le di semilla?
        if rng is None:
            rng = numpy.random.RandomState(1234)
        self.rng = rng

        self.task = (1 if task == "clasificacion" else 0)

        self.capas = []

        self.params = []

        self.cost   = 0.0
        self.L1     = 0.0
        self.L2_sqr = 0.0

        # para hacer el trakeo de la entrada en el grafo... self.x es el root de todo!!
        self.x = theano.tensor.matrix('x')

    def costos(self, y):
        """
        :param y: etiqueta de salida

        """
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically

        assert self.task, "Funcion solo valida para tareas de clasificacion"

        # costo, puede ser el MSE o bien el logaritmo negativo de la entropia..
        costo0 = self.capas[-1].negative_log_likelihood(y)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        costo1 = 0.0
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        costo2 = 0.0
        for capa in self.capas:
            costo1 += abs(capa.W.sum())
            costo2 += (capa.W ** 2).sum()

        self.cost   = costo0
        self.L1     = costo1
        self.L2_sqr = costo2



    def addLayer(self, unitsOut, classification, unitsIn=None, activation=Sigmoid(), weight=None, bias=None):
        if not self.capas:
            # primer capa, es la entrada de mlp, osea x, para iniciar el arbol
            entrada = self.x
        else:
            # la entrada es la salida de la ultima capa hasta ahora...
            entrada = self.capas[-1].activate()

        if unitsIn is None: # las entradas deben ser las salidas de la anterior
            assert self.capas != [], "Debe prover de unidades de entrada para esta capa."
            unitsIn = self.capas[-1].get_weights().get_value(borrow=True).shape[1]

        if not classification:
            capa = Layer(nIn = unitsIn,
                         nOut = unitsOut,
                         input = entrada,
                         rng = self.rng,
                         activationFn = activation,
                         W = weight,
                         b = bias)
        else:
            capa = ClassificationLayer(nIn = unitsIn,
                                       nOut = unitsOut,
                                       input = entrada,
                                       W = weight,
                                       b = bias)

        self.capas.append(capa)

        self.params += capa.params
        del capa

    def netErrors(self, y):
        return self.capas[-1].errors(y)

    def predict(self):
        assert self.task, "Funcion solo valida para tareas de clasificacion"
        return self.capas[-1].predict()


    def train(self, trainSet, validSet, testSet, batch_size,
        learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000):
        assert len(self.capas) != 0, "No hay capas cargadas en la red, <<addLayer()>>"

        # allocate symbolic variables for the data
        index = theano.tensor.lscalar() # index to a [mini]batch
        y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of
                                        # [int] labels

        #learning_rate=0.01; L1_reg=0.00; L2_reg=0.0001; n_epochs=1000

        trainX, trainY  = shared_dataset(trainSet)
        validX, validY  = shared_dataset(validSet)
        testX, testY    = shared_dataset(testSet)

        n_train_batches = trainX.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = validX.get_value(borrow=True).shape[0] // batch_size
        n_test_batches  = testX.get_value(borrow=True).shape[0] // batch_size

        # necesito actualizar los costos, si no hago este paso no tengo los valores requeridos
        self.costos(y)

        cost = (
                self.cost +
                L1_reg * self.L1 +
                L2_reg * self.L2_sqr)

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [theano.tensor.grad(cost, param) for param in self.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)
            ]

        # build functions
        test_model = theano.function(
                        inputs  = [index],
                        outputs = self.netErrors(y),
                        givens  = {
                            self.x: testX[index * batch_size:(index + 1) * batch_size],
                            y: testY[index * batch_size:(index + 1) * batch_size]
                        },
                        name = 'test_model'
        )

        validate_model = theano.function(
                        inputs  = [index],
                        outputs = self.netErrors(y),
                        givens = {
                            self.x: validX[index * batch_size:(index + 1) * batch_size],
                            y: validY[index * batch_size:(index + 1) * batch_size]
                        },
                        name = 'validate_model'
        )

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
                        inputs = [index],
                        outputs = cost,
                        updates = updates,
                        givens = {
                            self.x: trainX[index * batch_size: (index + 1) * batch_size],
                            y: trainY[index * batch_size: (index + 1) * batch_size]
                        },
                        name = 'train_model'
        )

        k = self.predict()
        ll = trainY
        predictor = theano.function(
                        inputs=[],
                        outputs=[k,ll],
                        givens={
                            self.x: trainX},
                        name='predictor'
        )

        print('... training')

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'epoca %i, error validacion %f %%' %
                        (
                            epoch,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoca %i, error test del modelo %f %%') %
                              (epoch, test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(('Optimizacion Completada. Mejor puntaje de validacion: %f %% '
               'con un performance en el test de %f %%') %
              (best_validation_loss * 100., test_score * 100.))
        print(('El codido del archivo ' +
               os.path.split(__file__)[1] +
               ' se ejecuto por %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

        print("reales", predictor()[1][0:10])
        print("predic", predictor()[0][0:10])

def load_dbn_weigths(path, dbnName):
    """
    carga las primeras 10 capas de la dbn segun su nombre y directorio (ordenadas por nombre)
    http://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered
    """
    import glob# load_dbn_weight
    capas = []
    for file in sorted(glob.glob(path + dbnName + "_capa[0-9].*")):
        print("Cargando capa: ",file) if verbose else None
        capas.append(rbm_gpu.load(str(file)))
    return capas

if __name__ == '__main__':
    assert False, str(__file__ + " No es un modulo")
