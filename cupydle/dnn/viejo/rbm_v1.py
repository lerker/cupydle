import numpy
import time

# TODO libro 55.pdf en Libros_RNA adaptacion de las arquitecturas profundas a problemas no estacionarios

# TODO
# - entrada binaria, unidades binarias visibles y ocultas

"""
Contrastive divergence is a recipe for training undirected graphical models (a class of probabilistic models used in machine learning). It relies on an approximation of the gradient (a good direction of change for the parameters) of the log-likelihood (the basic criterion that most probabilistic learning algorithms try to optimize) based on a short Markov chain (a way to sample from probabilistic models) started at the last example seen. It has been popularized in the context of Restricted Boltzmann Machines (Hinton & Salakhutdinov, 2006, Science), the latter being the first and most popular building block for deep learning algorithms. Its pseudo-code is very simple; you can see an example implementation in the deep learning tutorial there (in python):

http://deeplearning.net/tutorial...

You can find more mathematical details in numerous papers, starting with the above, or in the review paper / book that I wrote (sec. 5.4), downloadable from

http://www.iro.umontreal.ca/~ben...
"""

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
# END TIMER

class rbm(object):
    """Restricted Boltzmann Machine (RBM)
    % This program trains Restricted Boltzmann Machine in which
    % visible, binary, stochastic pixels are connected to
    % hidden, binary, stochastic feature detectors using symmetrically
    % weighted connections. Learning is done with 1-step Contrastive Divergence.
    % The program assumes that the following variables are set externally:
    % maxepoch  -- maximum number of epochs
    % numhid    -- number of hidden units
    % batchdata -- the data that is divided into batches (numcases numdims numbatches)
    % restart   -- set to 1 if learning starts from beginning
    """

    def __init__(
            self,
            n_visible=784,
            n_hidden=1000,
            w=None,
            hbias=None,
            vbias=None,
            numpy_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param n_visible: number of visible units

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

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.evolution = []

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if w is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            w = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=numpy.float32
            )

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = numpy.zeros(shape=(1, n_hidden), dtype=numpy.float32)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = numpy.zeros(shape=(1, n_visible), dtype=numpy.float32)

        self.w = w
        self.hbias = hbias
        self.vbias = vbias

        # Todo debe ser un parametro externo
        self.maxepoch = 15
        self.numcases = 100  # los numeros de casos son la cantidad de patrones en el bacth (filas)

    # END INIT

    def positive(self, data):
        positive_probabilidad_oculta_dada_visible = 1.0 / \
                                                    (1.0 + numpy.exp(
                                                        numpy.dot(-data, self.w) -
                                                        numpy.tile(self.hbias, (self.numcases, 1))
                                                    ))
        positive_prods = numpy.dot(data.T, positive_probabilidad_oculta_dada_visible)
        positive_oculta_activaciones = numpy.sum(a=positive_probabilidad_oculta_dada_visible, axis=0, dtype=numpy.float32)
        positive_visible_activaciones = numpy.sum(a=data, axis=0, dtype=numpy.float32)
        return positive_probabilidad_oculta_dada_visible, positive_prods, positive_oculta_activaciones, positive_visible_activaciones
    # END POSITIVE

    def negative(self, positive_hide_states):
        negative_data = 1.0 / (1.0 + numpy.exp(numpy.dot(-positive_hide_states, self.w.T) -
                                               numpy.tile(self.vbias, (self.numcases, 1))))
        negative_probabilidad_visible_dada_oculta = 1.0 / (1.0 + numpy.exp(numpy.dot(-negative_data, self.w) -
                                               numpy.tile(self.hbias, (self.numcases, 1))))
        negative_prods = numpy.dot(negative_data.T, negative_probabilidad_visible_dada_oculta)
        negative_oculta_activaciones = numpy.sum(a=negative_probabilidad_visible_dada_oculta, axis=0, dtype=numpy.float32)
        negative_visible_activaciones = numpy.sum(negative_data, axis=0, dtype=numpy.float32)
        return negative_data, negative_probabilidad_visible_dada_oculta, negative_prods, negative_oculta_activaciones, negative_visible_activaciones
    # END NEGATIVE

    def correr(self, input, maxepoch=15, numcases=100):

        self.maxepoch = maxepoch
        self.numcases = numcases

        vishidinc = numpy.zeros(shape=(self.n_visible, self.n_hidden), dtype=numpy.float32)
        hidbiasinc = numpy.zeros(shape=(1, self.n_hidden), dtype=numpy.float32)
        visbiasinc = numpy.zeros(shape=(1, self.n_visible), dtype=numpy.float32)

        ##
        ## print data information

        print("Training Dataset size: \t{}".format(input.shape[0]))
        print("Visible units # -> \t{}".format(self.n_visible))
        print("Hidden units # -> \t{}".format(self.n_hidden))
        print("Batch size # -> \t{}".format(self.numcases))
        print("Maxinium Epoch # -> \t{}".format(self.maxepoch))
        print("-------------------------------------------------------")
        ##

        for epoch in range(self.maxepoch):
            print('Starting Epoch {} of {}'.format(epoch, self.maxepoch))
            errorSum = 0.0
            # TODO hacer un for para cada bach, ahora supongo que tengo uno solo
            # for batch in range(size(databaches)):
            # data = databach(n)
            (tam, self.n_visible) = input.shape
            idx = range(0, tam+1, self.numcases)
            list_idx = list (zip (idx[0:-1] , idx[1:]) )
            for batch_idx in range(0, len(list_idx)):
                print('Epoch {} - batch {}'.format(epoch, batch_idx), end="\r")
                # comienzo fase positiva
                positive_probabilidad_oculta_dada_visible, \
                positive_prods, \
                positive_oculta_activaciones, \
                positive_visible_activaciones = self.positive(data=input[list_idx[batch_idx][0] : list_idx[batch_idx][1]])
                # fin fase positiva

                positive_ocultas_estados = positive_probabilidad_oculta_dada_visible > numpy.random.rand(self.numcases,
                                                                                                self.n_hidden)

                # comienzo fase negativa
                negative_data, \
                negative_probabilidad_visible_dada_oculta, \
                negative_prods, \
                negative_oculta_activaciones, \
                negative_visible_activaciones = self.negative(positive_hide_states=positive_ocultas_estados)
                # fin fase negativa

                err = numpy.sum(a=(numpy.sum(a=numpy.power( (input[list_idx[batch_idx][0] : list_idx[batch_idx][1]] - negative_data), 2), dtype=numpy.float32)), dtype=numpy.float32)
                errorSum = err + errorSum

                # TODO lo copio de hinton
                if epoch > self.maxepoch/2:
                    momentum = self.finalmomentum
                else:
                    momentum = self.initialmomentum
                ##


                # %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                vishidinc = momentum * vishidinc + self.epsilonw*( (positive_prods - negative_prods)/self.numcases - self.weightcost * self.w)

                visbiasinc = momentum * visbiasinc + (self.epsilonvb/self.numcases) * (positive_visible_activaciones - negative_visible_activaciones)

                hidbiasinc = momentum * hidbiasinc + (self.epsilonhb/self.numcases) * (positive_oculta_activaciones - negative_oculta_activaciones)

                self.w = self.w + vishidinc
                self.vbias = self.vbias + visbiasinc
                self.hbias = self.hbias + hidbiasinc

            print("Epoch {} - Error {}".format(str(epoch+1), errorSum))
            self.evolution.append(errorSum) # append de error each epoch


def binarize(d):
    return list(map(lambda x: x>=128, d))


def test_rbm():
    print("Testing RBM - MNIST!! 1-step Divergence Contrastive")

    ## cargar los datos
    from cupydle.test.mnist_loader import MNIST as mn
    m = mn(path='cupydle/data/')
    dataXtrn, dataYtrn = m.load_training()
    dataXtst, dataYtst = m.load_testing()

    # aca se copia tal cual el contenido en un array... hay que binarizarlo segun se necesite
    #dataXtrn = numpy.array(dataXtrn, dtype=numpy.float32)
    #dataYtrn = numpy.array(dataYtrn, dtype=numpy.float32)
    #dataXtst = numpy.array(dataXtst, dtype=numpy.float32)
    #dataYtst = numpy.array(dataYtst, dtype=numpy.float32)
    ##
    """ aca intente binarizar el contenido
    dataXtrn = numpy.array(list(map(binarize, dataXtrn)), dtype=numpy.int8)
    dataYtrn = numpy.array(list(map(binarize, dataYtrn)), dtype=numpy.int8)
    dataXtst = numpy.array(list(map(binarize, dataXtst)), dtype=numpy.int8)
    dataYtst = numpy.array(list(map(binarize, dataYtst)), dtype=numpy.int8)
    """

    aux = numpy.zeros_like(dataXtrn)

    for x in range(len(aux)):
        for i in range(len(dataXtrn[x])):
            if dataXtrn[x][i] >= 128:
                aux[x][i] = 1
            else:
                aux[x][i] = 0

    dataXtrn = numpy.array(aux, dtype=numpy.int8)


    n_visible = 784
    n_hidden = 1000
    numpy_rng = numpy.random.RandomState(1234)
    #datos = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
    #                                        high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
    #                                        size=(n_visible, n_hidden)), dtype=numpy.float32)

    r = rbm(n_visible=n_visible,n_hidden=n_hidden, numpy_rng=numpy_rng)

    start = time.time()
    r.correr(input=dataXtrn, maxepoch=10, numcases=100)
    end = time.time()

    print("Tiempo total: {}".format(timer(start,end)))

    ###### plot
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(r.evolution, 'ro')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title("Evolution" + " Hidden: " + str(n_hidden) + " Visible: " + str(n_visible) + " - Time: " + str(timer(start, end)))
    plt.show()
    fig.savefig("cupydle/data/rbm.png")


if __name__ == '__main__':
    test_rbm()
