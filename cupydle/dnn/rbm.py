#!/usr/bin/python3
import numpy


class rbm(object):

    def __init__(self, vec_x0, n_hidden=1000):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param vec_x0: vector of inputs

        :param n_hidden: number of hidden units

        :param w: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """
        self.epsilonw = 0.1  # Learning rate for weights
        self.epsilonvb = 0.1  # Learning rate for biases of visible units
        self.epsilonhb = 0.1  # Learning rate for biases of hidden units
        self.weightcost = 0.0002
        self.initialmomentum = 0.5
        self.finalmomentum = 0.9

        self.n_visible = len(vec_x0)
        self.n_hidden = n_hidden

        self.evolution = []

        # create a number generator
        numpy_rng = numpy.random.RandomState(1234)


        # W is initialized with `initial_W` which is uniformely
        # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
        # converted using asarray to dtype theano.config.floatX so
        # that the code is runable on GPU
        self.w = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    size=(self.n_visible, self.n_hidden)
                ),
                dtype=numpy.float32
            )


        self.hbias = numpy.zeros(shape=(1, self.n_hidden), dtype=numpy.float32)

        self.vbias = numpy.zeros(shape=(1, self.n_visible), dtype=numpy.float32)

        # Todo debe ser un parametro externo
        self.maxepoch = 15
        self.numcases = 100  # los numeros de casos son la cantidad de patrones en el bacth (filas)

    # END INIT


if __name__ == "__main__":

    entradas = numpy.zeros((1,784))
    rbm(entradas)
