#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implementation of restricted Boltzmann machine on GP-GPU."""

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"

## TODO
### las versiones de binomial... para la GPU
# http://deeplearning.net/software/theano/tutorial/examples.html#example-other-random
# There are 2 other implementations based on MRG31k3p and CURAND.
# The RandomStream only work on the CPU, MRG31k3p work on the CPU and GPU. CURAND only work on the GPU.

# sistema basico
import sys  # llamadas al sistema en errores
from subprocess import call # para ejecutar programas
import numpy

### THEANO
# TODO mejorar los imports, ver de agregar theano.shared... etc
import theano
#from theano.tensor.shared_randomstreams import RandomStreams  #random seed CPU
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams # GPU

## ultra rapido
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU - GPU
                                                                        #(parece que binomial no esta implementado, lo reemplaza por uniform)
                                                                        # cambiar a: multinomial(size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None)[source]
                                                                        # en activationFunction

theanoFloat  = theano.config.floatX

from cupydle.dnn.activations import Sigmoid


"""




"""
from cupydle.dnn.utils import timer as timer2
try:
    import PIL.Image as Image
except ImportError:
    import Image

from cupydle.dnn.utils import scale_to_unit_interval
from cupydle.dnn.utils import tile_raster_images

import matplotlib.pyplot

class RBM(object):
    """Restricted Boltzmann Machine on GP-GPU (RBM-GPU)  """

    def __init__(
                self,
                n_visible=784,      # cantidad de neuronas visibles
                n_hidden=500,       # cantidad de neuronas ocultas
                w=None,             # matriz de pesos
                visbiases=None,     # vector de biases visibles
                hidbiases=None,     # vector de biases ocultos
                numpy_rng=None,     # seed para numpy random
                theano_rng=None):   #seed para theano random
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param w: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hidbiases: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param visbiases: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible      = n_visible
        self.n_hidden       = n_hidden

        # create a number generator (fixed) for test NUMPY
        if numpy_rng is None:
            # TODO cambiar a aleatorio
            numpy_rng = numpy.random.RandomState(1234)

        # create a number generator (fixed) for test THEANO
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            # TODO cambiar a aleatorio
            theano_rng = RandomStreams(seed=1234)

        # TODO agregar agregar el borrow
        if w is None:
            # w is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_visible+n_hidden)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            _w = numpy.asarray(
                    numpy_rng.uniform(
                        low= -4 * numpy.sqrt(6. / (self.n_visible + self.n_hidden)),
                        high= 4 * numpy.sqrt(6. / (self.n_visible + self.n_hidden)),
                        size=(self.n_visible, self.n_hidden)),
                    dtype=theanoFloat)

            # pass the buffer to theano namespace (GPU)
            w = theano.shared(value=_w, name='w', borrow=True)
            del _w

        # create shared variable for hidden units bias
        if hidbiases is None:
            #_hidbiases = numpy.zeros(shape=(1, self.n_hidden),
            #                         dtype=theanoFloat)
            _hidbiases = numpy.zeros(shape=(self.n_hidden),
                                     dtype=theanoFloat)
            hidbiases = theano.shared(value=_hidbiases, name='hidbiases', borrow=True)
            del _hidbiases


        # create shared variable for visible units bias
        if visbiases is None:
            #_visbiases = numpy.zeros(shape=(1, self.n_visible),
            #                         dtype=theanoFloat)
            _visbiases = numpy.zeros(shape=(self.n_visible),
                                     dtype=theanoFloat)
            visbiases = theano.shared(value=_visbiases, name='visbiases', borrow=True)
            del _visbiases


        if w is not None and isinstance(w, numpy.ndarray):
            w = theano.shared(value=w, name='w', borrow=True)

        ########
        self.visbiases  = visbiases
        self.hidbiases  = hidbiases
        self.w          = w
        self.numpy_rng  = numpy_rng
        self.theano_rng = theano_rng

        # funcion de activacion sigmodea para cada neurona (porbabilidad->binomial)
        self.activationFunction = Sigmoid()

        # buffers para el almacenamiento temporal de las variables
        # momento-> incrementos, historicos
        self.vishidinc   = theano.shared(value=numpy.zeros(shape=(self.n_visible, self.n_hidden), dtype=theanoFloat), name='vishidinc')
        self.hidbiasinc  = theano.shared(value=numpy.zeros(shape=(self.n_hidden), dtype=theanoFloat), name='hidbiasinc')
        self.visbiasinc  = theano.shared(value=numpy.zeros(shape=(self.n_visible), dtype=theanoFloat), name='visbiasinc')

        # para las derivadas
        self.internalParams = [self.w, self.hidbiases, self.visbiases]

        self.x = theano.tensor.matrix(name="x")   # root

        self.params = {}
        self._initParams()

        # for allocate statistics
        self.estadisticos = {}
        self._initStatistics()
    # END INIT

    def _initParams(self):
        """ inicializa los parametros de la red, un diccionario"""
        self.params['epsilonw'] = 0.0
        self.params['epsilonvb'] = 0.0
        self.params['epsilonhb'] = 0.0
        self.params['weightcost'] = 0.0
        self.params['initialmomentum'] = 0.0
        self.params['finalmomentum'] = 0.0
        self.params['maxepoch'] = 0.0
        self.params['activationfuntion'] = Sigmoid()
        return 1

    def set_w(self, w):
        #if isinstance(w, numpy.ndarray):
        #    self.w.set_value(w)
        self.w.set_value(w)
        return 1

    def set_biasVisible(self, bias):
        self.visbiases.set_value(bias)
        return 1

    def set_biasOculto(self, bias):
        self.hidbiases.set_value(bias)
        return 1

    def get_w(self):
        return self.w.get_value()

    def get_biasVisible(self):
        return self.visbiases.get_value()

    def get_biasOculto(self):
        return self.hidbiases.get_value()

    def printParams(self):
        for key, value in self.params.items():
            print('{:>20}: {:<10}'.format(str(key), str(value)))
        return 1

    def setParams(self, parametros):
        if not isinstance(parametros, dict):
            assert False, "necesito un diccionario"

        for key, _ in parametros.items():
            if key in self.params:
                self.params[key] = parametros[key]
            else:
                assert False, "la clave(" + str(key) + ") en la variable paramtros no existe"

        return 1

    def _initStatistics(self):
        # incializo el diccionario con los estadisticos
        dic = {
                'errorEntrenamiento':[],
                'errorValidacion':[],
                'errorValidacion':[],
                'mseEntrenamiento':[],
                'errorTesteo':[],
                'energiaLibreEntrenamiento':[],
                'energiaLibreValidacion': []
                }

        self.estadisticos = dic
        del dic
        return 1

    def agregarEstadistico(self, estadistico):
        # cuando agrega un estadistico (ya sea uno solo) se considera como una epoca nueva
        # por lo tanto se setea el resto de los estadisitcos con 0.0 los cuales no fueron
        # provistos

        if not isinstance(estadistico, dict):
            assert False, str(repr('estadistico') + " debe ser un tipo diccionario")

        # copio para manipular sin cambiar definitivamente
        viejo = self.estadisticos

        for key, _ in estadistico.items():
            bandera=True
            if key in viejo:
                viejo[key] += [estadistico[key]]
                bandera = False
            else:
                assert False, str("No exite el key " + repr(key))
            if bandera:
                viejo[key]+=[0.0]

        self.estadisticos = viejo
        del viejo
        return


    def plot_weigth(self, weight=None, save=None, path=None):
        """
        Grafica la matriz de pesos de (n_visible x n_ocultas) unidades

        :param weight: matriz de pesos asociada a una RBM, cualquiera
        """
        import matplotlib
        import matplotlib.pyplot as plt

        if weight is None:
            weight = self.w.get_value()

        if not type(weight) == numpy.array:
            weight = numpy.asarray(weight)

        # convert a vector array (weight) to matrix sample => (self.n_visible, self.n_hidden)
        if weight.shape == (self.n_visible + self.n_hidden,): # se recorre con un solo indice, i=#
            weight = numpy.reshape(weight, (self.n_visible, self.n_hidden))
        elif weight.shape == (self.n_visible + self.n_hidden,1): # se recorre la weightn con dos indices, j=0
            weight = numpy.reshape(weight, (self.n_visible, self.n_hidden))
        elif weight.shape == (self.n_visible, self.n_hidden):
            pass
        else:
            sys.exit("No se reconoce la dimesion de la matriz de pesos")

        """
        fig = plt.figure('Pesos')
        ax = fig.add_subplot(2, 1, 1)
        ax.matshow(weight, cmap = matplotlib.cm.binary)
        plt.xticks(numpy.array([]))
        plt.yticks(numpy.array([]))
        #plt.title(label)
        """
        fig, ax = plt.subplots()
        cax = ax.imshow(weight, interpolation='nearest', cmap=matplotlib.cm.binary)
        plt.xticks([0,weight.shape[1]/4, weight.shape[1]/2, 3*weight.shape[1]/4, weight.shape[1]])
        plt.yticks([0,weight.shape[0]/4, weight.shape[0]/2, 3*weight.shape[0]/4, weight.shape[0]])
        plt.xlabel('# Hidden Neurons')
        plt.ylabel('# Visible Neurons')
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax)
        cbar.ax.set_yticklabels(['0', '','','','','','','','1']) # el color bar tiene 10 posiciones para los ticks

        plt.title('Weight Matrix')

        if save is not None and path is None:   # corrigo la direccion en caso de no proporcionarla y si guardar
            path = ''
        if save == 'png' or save is True:
            plt.savefig(path + "weightMatrix" + ".png", format='png')
        elif save == 'eps':
            plt.savefig(path + 'weightMatrix' + '.eps', format='eps', dpi=1000)
        elif save == 'svg':
            plt.savefig(path + 'weightMatrix' + '.svg', format='svg', dpi=1000)
        else:
            pass

        plt.show()

        return 1

    def plot_error(self, error, tipo='validation', show=True, save=None, path=None):
        # importo aca para no sobrecargar.. si no es necesario
        import matplotlib
        import matplotlib.pyplot as plt

        if type(error) is not dict and type(error) is not list:
            print("No se reconoce el contenedor del error")
            sys.exit(0)


        fig = plt.figure('Error').clear()

        # si el error viene en forma de lista considero un error/epoca
        # si el error viene en un diccionario, considero que es el error de validacion
        # el que quiero plotear, caso contrario debo especificar en el parametro 'tipo'
        if type(error) == dict:
            if tipo == 'validation':
                y = error['errorValidation']
            elif tipo == 'training':
                y = error['errorTraining']
            else:
                print("No se reconoce el parametro 'tipo'")
                sys.exit(0)
        elif type(error) == list:
            y = error

        if tipo == 'validation':
            ylabel = 'Validation Set Error'
        elif tipo == 'training':
            ylabel = 'Training Set Error'
        else:
            print("No se reconoce el parametro 'tipo'")
            sys.exit(0)

        if tipo == 'validation':
            marker = 'or'   # bolita roja
        else:
            marker = 'og'   # bolita verde

        xlabel = 'Epoch'
        x = list(range(1, len(y)+1))

        plt.plot(x, error, marker)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(ylabel)

        if show:
            plt.show()


        if save is not None and path is None:   # corrigo la direccion en caso de no proporcionarla y si guardar
            path = ''
        if save == 'png' or save is True:
            route = path + ylabel.replace(' ', '_') + ".png"
            plt.savefig(route, format='png')
            print('Guardando la imagen en la ruta:',route)
        elif save == 'eps':
            route = path + ylabel.replace(' ', '_') + '.eps'
            plt.savefig(route, format='eps', dpi=1000)
            print('Guardando la imagen en la ruta:',route)
        elif save == 'svg':
            route = path + ylabel.replace(' ', '_') + '.svg'
            plt.savefig(route, format='svg', dpi=1000)
            print('Guardando la imagen en la ruta:',route)
        else:
            pass

        return plt

    def markovChain(self, steps):
        """
        Ejecuta una cadena de Markov de 'k=steps' pasos

        :param steps: nodo escalar de la cantidad de pasos

        La Cadena de Markov comienza del ejemplo visible, luego se samplea
        la unidad oculta, el siguiente paso es samplear las unidades visibles
        a partir de la oculta (reconstruccion)
        """
        # un paso de CD es V->H->V
        def oneStep(vsample, wG, vbiasG, hbiasG):
            # positive
            # theano se da cuenta y no es necesario realizar un 'theano.tensor.tile' del
            # bias (dimension), ya que lo hace automaticamente
            linearSumP                  = theano.tensor.dot(vsample, wG) + hbiasG
            hiddenActData, probabilityP = self.activationFunction.nonDeterminstic(linearSumP)
            #negative
            linearSumN                  = theano.tensor.dot(hiddenActData, wG.T) + vbiasG
            visibleActRec, probabilityN = self.activationFunction.nonDeterminstic(linearSumN)
            return [visibleActRec, hiddenActData, probabilityP, probabilityN, linearSumP, linearSumN]

        # loop o scan, devuelve los tres valores de las funcion y un diccionario de 'actualizaciones'
        ( [visibleActRec, hiddenActData, probabilityP, probabilityN, linearSumP, linearSumN],
          updates) = theano.scan(   fn           = oneStep,
                                    outputs_info = [self.x, None, None, None, None, None],
                                    non_sequences= [self.w, self.visbiases, self.hidbiases],
                                    n_steps      = steps,
                                    strict       = True,
                                    name         = 'scan_oneStepMarkovVHV')
        return ([visibleActRec[-1], hiddenActData[-1], probabilityP[-1], probabilityN[-1], linearSumP[-1], linearSumN[-1]], updates)

    def energiaLibre(self, vsample):
        ''' Function to compute the free energy '''
        wx_b = theano.tensor.dot(vsample, self.w) + self.hidbiases
        vbias_term = theano.tensor.dot(vsample, self.visbiases)
        hidden_term = theano.tensor.sum(theano.tensor.log(1 + theano.tensor.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def crearDibujo(self, datos, axe=None, titulo='', estilo='b-'):
        #debo asignar la vuelta de axe al axes del dibujo, solo que lo pase por puntero que todavia no se

        # linestyle or ls   [ '-' | '--' | '-.' | ':' | 'steps' | ...]
        #marker  [ '+' | ',' | '.' | '1' | '2' | '3' | '4' ]
        if axe is None:
            axe = matplotlib.pyplot.gca()

        ejeXepocas = range(1,len(datos)+1)
        axe.plot(ejeXepocas, datos, estilo)
        axe.set_title(titulo)
        axe.set_xticks(ejeXepocas) # los ticks del eje de las x solo en los enteros
        return axe

    def dibujarEstadisticos(self, show=False, save=None):

        errorEntrenamiento = self.estadisticos['errorEntrenamiento']
        errorValidacion = self.estadisticos['errorValidacion']
        errorTesteo = self.estadisticos['errorTesteo']
        mseEntrenamiento = self.estadisticos['mseEntrenamiento']
        energiaLibreEntrenamiento = self.estadisticos['energiaLibreEntrenamiento']
        energiaLibreValidacion = self.estadisticos['energiaLibreValidacion']

        f, axarr = matplotlib.pyplot.subplots(2, 3, sharex='col', sharey='row')

        axarr[0, 0] = self.crearDibujo(errorEntrenamiento, axarr[0, 0], titulo='Error de Entrenamiento', estilo='r-')
        axarr[0, 1] = self.crearDibujo(errorValidacion, axarr[0, 1], titulo='Error de Validacion', estilo='b-')
        axarr[0, 2] = self.crearDibujo(errorTesteo, axarr[0, 2], titulo='Error de Testeo')

        axarr[1, 0] = self.crearDibujo(mseEntrenamiento, axarr[1, 0], titulo='MSE de Entrenamiento')
        axarr[1, 1] = self.crearDibujo(energiaLibreEntrenamiento, axarr[1, 1], titulo='Energia Libre Entrenamiento')
        axarr[1, 2] = self.crearDibujo(energiaLibreValidacion, axarr[1, 2], titulo='Energia Libre Validacion')

        matplotlib.pyplot.tight_layout()

        if save is not None:
            print("guardando los estadisticos en: " + save)
            matplotlib.pyplot.savefig(save, bbox_inches='tight')
            #matplotlib.pyplot.savefig("estadisticos.png")

        if show:
            matplotlib.pyplot.show()
        return

    def dibujarFiltros(self, nombreArchivo, ruta, binary=False):
        # plot los filtros iniciales (sin entrenamiento)
        image = Image.fromarray(
            tile_raster_images(
                X=self.w.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        if binary:
            gray = image.convert('L')
            # Let numpy do the heavy lifting for converting pixels to pure black or white
            bw = numpy.asarray(gray).copy()
            # Pixel range is 0...255, 256/2 = 128
            bw[bw < 128] = 0    # Black
            bw[bw >= 128] = 255 # White
            # Now we put it back in Pillow/PIL land
            image = Image.fromarray(bw)

        image.save(ruta + nombreArchivo)

        return 1

    def pseudoLikelihoodCost(self, updates=None):
        # pseudo-likelihood is a better proxy for PCD
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = theano.tensor.round(self.x)

        # calculate free energy for the given bit configuration
        fe_xi = self.energiaLibre(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = theano.tensor.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.energiaLibre(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = theano.tensor.mean(self.n_visible \
             * theano.tensor.log(theano.tensor.nnet.sigmoid(fe_xi_flip
                                                            - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        # TODO es necesario retornar las updates?
        # ni siquiera las utiliza en el algoritmo.. medio inutil pasarla o retornarla
        return cost

    def PersistentConstrastiveDivergence(self, miniBatchSize, sharedData):

        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((miniBatchSize, self.n_hidden),
                                                     dtype=theanoFloat),
                                         borrow=True)

        # un paso de CD es H->V->H
        def oneStep(hsample, wG, vbiasG, hbiasG):
            #negative
            linearSumN                  = theano.tensor.dot(hsample, wG.T) + vbiasG
            visibleActRec, probabilityN = self.activationFunction.nonDeterminstic(linearSumN)
            # positive
            linearSumP                  = theano.tensor.dot(visibleActRec, wG) + hbiasG
            hiddenActData, probabilityP = self.activationFunction.nonDeterminstic(linearSumP)
            return [hiddenActData, visibleActRec, probabilityP, probabilityN, linearSumP, linearSumN]

        ( [hiddenActData, visibleActRec, probabilityP, probabilityN, linearSumP, linearSumN],
          updates) = theano.scan(   fn           = oneStep,
                                    outputs_info = [persistent_chain, None, None, None, None, None],
                                    non_sequences= [self.w, self.visbiases, self.hidbiases],
                                    n_steps      = steps,
                                    strict       = True,
                                    name         = 'scan_oneStepHVH')


        chain_end = visibleActRec[-1]

        cost = theano.tensor.mean(self.energiaLibre(self.x)) - theano.tensor.mean(
            self.energiaLibre(chain_end))

        deltaEnergia = cost

        # We must not compute the gradient through the gibbs sampling
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[chain_end])
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.internalParams):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * theano.tensor.cast(
                self.params['epsilonw'], dtype=theanoFloat
            )

        # Note that this works only if persistent is a shared variable
        updates[persistent_chain] = hiddenActData[-1]


        monitoring_cost = self.pseudoLikelihoodCost(updates)

        errorCuadratico = self.reconstructionCost_MSE(chain_end)

        train_rbm = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[monitoring_cost, errorCuadratico, deltaEnergia],
                        updates=updates,
                        givens={
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
                        name='train_rbm_pcd'
        )

        return train_rbm

    def reconstructionCost(self, linearSumN):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """
        # reconstruction cross-entropy is a better proxy for CD
        crossEntropy = theano.tensor.mean(
            theano.tensor.sum(
                self.x * theano.tensor.log(theano.tensor.nnet.sigmoid(linearSumN)) +
                (1 - self.x) * theano.tensor.log(1 - theano.tensor.nnet.sigmoid(linearSumN)),
                axis=1
            )
        )
        return crossEntropy

    def reconstructionCost_MSE(self, reconstrucciones):
        # mean squared error, error cuadratico medio

        mse = theano.tensor.mean(
                theano.tensor.sum(
                    theano.tensor.sqr( self.x - reconstrucciones ), axis=1
                )
        )

        return mse

    def ConstrastiveDivergence(self, miniBatchSize, sharedData):
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # un paso de CD es V->H->V
        def oneStep(vsample, wG, vbiasG, hbiasG):
            # positive
            # theano se da cuenta y no es necesario realizar un 'theano.tensor.tile' del
            # bias (dimension), ya que lo hace automaticamente
            linearSumP                  = theano.tensor.dot(vsample, wG) + hbiasG
            hiddenActData, probabilityP = self.activationFunction.nonDeterminstic(linearSumP)
            #negative
            linearSumN                  = theano.tensor.dot(hiddenActData, wG.T) + vbiasG
            visibleActRec, probabilityN = self.activationFunction.nonDeterminstic(linearSumN)
            return [visibleActRec, hiddenActData, probabilityP, probabilityN, linearSumP, linearSumN]

        # loop o scan, devuelve los tres valores de las funcion y un diccionario de 'actualizaciones'
        ( [visibleActRec, hiddenActData, probabilityP, probabilityN, linearSumP, linearSumN],
          updates) = theano.scan(   fn           = oneStep,
                                    outputs_info = [self.x, None, None, None, None, None],
                                    non_sequences= [self.w, self.visbiases, self.hidbiases],
                                    n_steps      = steps,
                                    strict       = True,
                                    name         = 'scan_oneStepVHV')

        chain_end = visibleActRec[-1]

        cost = theano.tensor.mean(self.energiaLibre(self.x)) - theano.tensor.mean(
            self.energiaLibre(chain_end))

        deltaEnergia = cost

        # We must not compute the gradient through the gibbs sampling
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[chain_end])
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.internalParams):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * theano.tensor.cast(
                self.params['epsilonw'], dtype=theanoFloat
            )


        monitoring_cost = self.reconstructionCost(linearSumN[-1])

        errorCuadratico = self.reconstructionCost_MSE(chain_end)

        train_rbm = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[monitoring_cost, errorCuadratico, deltaEnergia],
                        updates=updates,
                        givens={
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_cd'
        )

        return train_rbm

    def train(self, data, miniBatchSize=10, pcd=True, gibbsSteps=1, validationData=None, plotFilters=None, printCompacto=False):

        print("Entrenando una RBM, con [{}] unidades visibles y [{}] unidades ocultas".format(self.n_visible, self.n_hidden))
        print("Cantidad de ejemplos para el entrenamiento no supervisado: ", len(data))

        # convierto todos los datos a una variable shared de theano para llevarla a la GPU
        sharedData  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='TrainingData')

        # para la validacion
        if validationData is not None:
            sharedValidationData = theano.shared(numpy.asarray(a=validationData, dtype=theanoFloat), name='ValidationData')


        trainer = None

        if pcd:
            print("Entrenando con Divergencia Contrastiva Persistente.")
            trainer = self.PersistentConstrastiveDivergence(miniBatchSize, sharedData)
        else:
            print("Entrenando con Divergencia Contrastiva.")
            trainer = self.ConstrastiveDivergence(miniBatchSize, sharedData)

        if plotFilters is not None:
            # plot los filtros iniciales (sin entrenamiento)
            self.dibujarFiltros(  nombreArchivo='filtros_epoca_0.pdf',
                                ruta=plotFilters)

        # cantidad de indices... para recorrer el set
        indexCount = int(data.shape[0]/miniBatchSize)
        costo = numpy.Inf
        mse = numpy.Inf
        fEnergy = numpy.Inf
        finLinea='\n'
        finLinea = '\r' if printCompacto else '\n'

        for epoch in range(0, self.params['maxepoch']):
            # imprimo algo de informacion sobre la terminal
            print(str('Epoca {:>3d} '
                    + 'de {:>3d}, '
                    + 'error<TrnSet>:{:> 8.5f}, '
                    + 'MSE<ejemplo> :{:> 8.5f}, '
                    + 'EnergiaLibre<ejemplo>:{:> 8.5f}').format(
                        epoch+1,
                        self.params['maxepoch'],
                        costo,
                        mse,
                        fEnergy),
                    end=finLinea)

            costo = []
            mse = []
            fEnergy = []
            for batch in range(0, indexCount):
                # salida[monitoring_cost, mse, deltafreeEnergy]
                salida = trainer(batch, gibbsSteps)

                costo.append(salida[0])
                mse.append(salida[1])
                fEnergy.append(salida[2])


            costo = numpy.mean(costo)
            mse = numpy.mean(mse)
            fEnergy = numpy.mean(fEnergy)

            self.agregarEstadistico(
                {'errorEntrenamiento': costo,
                 'mseEntrenamiento': mse,
                 'errorValidacion': 0.0,
                 'errorTesteo': 0.0,
                 'energiaLibreEntrenamiento': fEnergy,
                 'energiaLibreValidacion': 0.0})

            if plotFilters is not None:
                self.dibujarFiltros(nombreArchivo='filtros_epoca_{}.pdf'.format(epoch+1),
                                  ruta=plotFilters)

            # END SET
        # END epoch
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior
        return 1

    def reconstruccion(self, vsample, gibbsSteps=1):
        """
        realiza la reconstruccion a partir de un ejemplo, efectuando una cadena
        de markov con x pasos de gibbs sobre la misma

        puedo pasarle un solo ejemplo o una matrix con varios de ellos por fila
        """

        if vsample.ndim == 1:
            # es un vector, debo cambiar el root 'x' antes de armar el grafo
            # para que coincida con la entrada
            viejoRoot = self.x
            self.x = theano.tensor.fvector('x')

        data  = theano.shared(numpy.asarray(a=vsample,dtype=theanoFloat), name='datoReconstruccion')

        # realizar la cadena de markov k veces
        (   [visibleActRec, # actualiza la cadena
            _,
            _,
            probabilityN,
            _,
            _], # linear sum es la que se plotea
            updates        ) = self.markovChain(gibbsSteps)

        reconstructor = theano.function(
                        inputs=[],
                        outputs=[probabilityN, visibleActRec],
                        updates=updates,
                        givens={self.x: data},
                        name='reconstructor'
        )

        salida = reconstructor()[0]

        if vsample.ndim == 1:
            # hago el swap
            self.x = viejoRoot

        return salida

    def activacionesOcultas(self, vsample, gibbsSteps=1):
        """
        retornar las activaciones de las ocultas, para las dbn
        """

        if vsample.ndim == 1:
            # es un vector, debo cambiar el root 'x' antes de armar el grafo
            # para que coincida con la entrada
            viejoRoot = self.x
            self.x = theano.tensor.fvector('x')

        data  = theano.shared(numpy.asarray(a=vsample,dtype=theanoFloat), name='datoReconstruccion')

        # realizar la cadena de markov k veces
        (   [_, # actualiza la cadena
            hiddenActData,
            _,
            _,
            _,
            _], # linear sum es la que se plotea
            updates        ) = self.markovChain(gibbsSteps)

        activador = theano.function(
                        inputs=[],
                        outputs=[hiddenActData],
                        updates=updates,
                        givens={self.x: data},
                        name='activador'
        )

        salida = activador()[0]

        if vsample.ndim == 1:
            # hago el swap
            self.x = viejoRoot

        return salida

    def sampleo(self, data, labels=None, chains=20, samples=10, gibbsSteps=1000,
                patchesDim=(28,28), binary=False):
        """
        Realiza un 'sampleo' de los datos con los parametros 'aprendidos'
        El proposito de la funcion es ejemplificar como una serie de ejemplos
        se muestrean a traves de la red ejecuntado sucesivas cadenas de markov

        :type data: numpy.narray (sample, data)
        :param data: conjunto de datos a extraer las muestras

        :type labels: numpy.narray (sample,)
        :param labels: conjunto de etiquetas que se corresponden a los datos

        :type chains: int
        :param chains: cantidad de cadenas parallelas de Gibbs de las cuales se muestrea

        :type samples: int
        :param samples: cantidad de muestras a realizar sobre cada cadena

        :type gibbsSteps: int
        :param gibssSteps: catidad de pasos de gibbs a ejecutar por cada muestra

        :type patchesDim: tuple ints
        :param patchesDim: dimesion de los patches 'sampleo' de cada cadena,
                            (alto, ancho). Para mnist (28,28)

        :type binary: bool
        :param binary: la imagen de salida debe ser binarizada
        """

        data  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='DataSample')
        n_samples = data.get_value(borrow=True).shape[0]

        # seleeciona de forma aleatoria donde comienza a extraer los ejemplos
        # devuelve un solo indice.. desde [0,..., n - chains]
        test_idx = self.numpy_rng.randint(n_samples - chains)

        # inicializacion de la cadena, estado que persiste
        cadenaFija = theano.shared(
                        numpy.asarray(data.get_value(borrow=True)
                                        [test_idx:test_idx + chains],
                                    dtype=theanoFloat
                        )
        )
        # tengo leyenda sobre la imagen?
        if labels is not None:
            lista = range(test_idx, test_idx + chains)
            print("labels: ", str(labels[lista]))

        # realizar la cadena de markov k veces
        (   [visibleActRec, # actualiza la cadena
            _,
            _,
            probabilityN,
            _,
            _], # linear sum es la que se plotea
            updates        ) = self.markovChain(gibbsSteps)

        # cambiar el valor de la cadenaFija al de la reconstruccion de las visibles
        updates.update({cadenaFija: visibleActRec})

        # funcion princial
        muestreo = theano.function(
                        inputs=[],
                        outputs=[probabilityN, visibleActRec],
                        updates=updates,
                        givens={self.x: cadenaFija},
                        name='muestreo'
        )

        # dimensiones de los patches, para el mnist es (28,28)
        #ancho=28
        #alto=28
        alto, ancho = patchesDim
        imageResults = numpy.zeros(((alto+1) * samples + 1, (ancho+1) * chains - 1),
                            dtype='uint8')
        for idx in range(samples):
            #genero muestras y visualizo cada gibssSteps, ya que las muestras intermedias
            # estan muy correlacionadas, se visializa la probabilidad de activacion de
            # las unidades ocultas (NO la muestra binomial)
            probabilityN, visiblerecons = muestreo()

            print(' ... plotting sample {}'.format(idx))
            imageResults[(alto+1) * idx:(ancho+1) * idx + ancho, :] \
                = tile_raster_images(X=probabilityN,
                                    img_shape=(alto, ancho),
                                    tile_shape=(1, chains),
                                    tile_spacing=(1, 1)
                )

        # construct image
        image = Image.fromarray(imageResults)

        nombreArchivo = "samples_" + str(labels[lista])
        nombreArchivo = nombreArchivo.replace(" ", "_")

        # poner las etiquetas sobre la imagen
        watermark = False
        if watermark:
            from PIL import ImageDraw, ImageFont
            # get the ImageDraw item for this image
            draw = ImageDraw.Draw(image)
            fontsize = 15
            font = ImageFont.truetype("arial.ttf", fontsize)
            #fill = (255,255,255) # blanco
            fill = (255,255,0) # amarillo
            fill = 255
            draw.text((0, 0),text=str(labels[lista]),fill=fill,font=font)

        if binary:
            gray = image.convert('L')
            # Let numpy do the heavy lifting for converting pixels to pure black or white
            bw = numpy.asarray(gray).copy()
            # Pixel range is 0...255, 256/2 = 128
            bw[bw < 128] = 0    # Black
            bw[bw >= 128] = 255 # White
            # Now we put it back in Pillow/PIL land
            image2 = Image.fromarray(bw)
            image2.save(nombreArchivo + '_binary.pdf')

        image.save(nombreArchivo + '.pdf')

        return 1

    def buildUpdates(self, updates, cost, constant):
        # TODO aca le quite le constante (en el tutorial estaba), parece que anda pero no se, preferia dejarlo
        # parece que anda mucho mas lento... si lo saco
        # We must not compute the gradient through the gibbs sampling
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[constant])
        #gparams = theano.tensor.grad(cost, self.internalParams)

        # creo una lista con las actualizaciones viejas (shared guardadas)
        oldUpdates = [self.vishidinc, self.hidbiasinc, self.visbiasinc]

        # una lista de ternas [(w,dc/dw,w_old),(hb,dc/dhb,...),...]
        parametersTuples = zip(self.internalParams, gparams, oldUpdates)

        momentum = self.params['initialmomentum']
        otherUpdates = []

        for param, delta, oldUpdate in parametersTuples:
            # segun el paramerto tengo diferentes learning rates
            if param.name == self.w.name:
                paramUpdate = momentum * oldUpdate \
                            - theano.tensor.cast( self.params['epsilonw'],
                                dtype=theanoFloat) * delta
            if param.name == self.visbiases.name:
                paramUpdate = momentum * oldUpdate \
                            - theano.tensor.cast(self.params['epsilonvb'],
                                dtype=theanoFloat) * delta
            if param.name == self.hidbiases.name:
                paramUpdate = momentum * oldUpdate \
                            - theano.tensor.cast(self.params['epsilonhb'],
                                dtype=theanoFloat) * delta

            newParam = param + paramUpdate
            otherUpdates.append((param, newParam))
            otherUpdates.append((oldUpdate, paramUpdate))

        updates.update(otherUpdates)
        return updates

    def buildUpdates2(self, updates, cost, constant):
        # TODO en la version1 agregue el momento,.. ver si anda
        # We must not compute the gradient through the gibbs sampling
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[constant])

        # TODO add Momentum
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.internalParams):
            # make sure that the learning rate is of the right dtype
            if param.name == self.w.name:
                updates[param] = (param
                                  - gparam
                                  * theano.tensor.cast(self.params['epsilonw'],
                                        dtype=theanoFloat)
                                  - param
                                  * theano.tensor.cast(self.params['weightcost'],
                                        dtype=theanoFloat))
            if param.name == self.visbiases.name:
                updates[param] = (param
                                  - gparam
                                  * theano.tensor.cast(self.params['epsilonvb'],
                                        dtype=theanoFloat))
            if param.name == self.hidbiases.name:
                updates[param] = (param
                                  - gparam
                                  * theano.tensor.cast(self.params['epsilonhb'],
                                        dtype=theanoFloat))
        return updates



    def save(self, nombreArchivo=None, method='simple', compression=None, layerN=0, absolutName=False):
        """
        guarda a disco la instancia de la RBM
        method is simple, no guardo como theano
        :param nombreArchivo:
        :param compression:
        :param layerN: numero de capa a la cual pertence la rbm (DBN)
        :param absolutName: si es true se omite los parametros a excepto el nombreArchivo
        :return:
        """
        #from cupydle.dnn.utils import save as saver

        #if nombreArchivo is not set
        if absolutName is False:
            if nombreArchivo is None:
                nombreArchivo = "V_" + str(self.n_visible) + "H_" + str(self.n_hidden) + "_" + time.strftime('%Y%m%d_%H%M')
            else:
                nombreArchivo = nombreArchivo + "_" + time.strftime('%Y-%m-%d_%H%M')

            if layerN != 0:
                nombreArchivo = "L_" + str(layerN) + nombreArchivo
        else:
            nombreArchivo = nombreArchivo

        if method != 'theano':
            from cupydle.dnn.utils import save as saver
            saver(objeto=self, filename=nombreArchivo, compression=compression)
        else:
            with open(nombreArchivo + '.zip','wb') as f:
                # arreglar esto
                from cupydle.dnn.utils_theano import save as saver
                saver(objeto=self, filename=nombreArchivo, compression=compression)

        return
    # END SAVE

    @staticmethod
    def load(nombreArchivo=None, method='simple', compression=None):
        """
        Carga desde archivo un objeto RBM guardado. Se distinguen los metodos
        de guardado, pickle simple o theano.

        :type nombreArchivo: string
        :param nombreArchivo: ruta completa al archivo donde se aloja la RBM

        :type method: string
        :param method: si es 'simple' se carga el objeto con los metodos estandar
                        para pickle, si es 'theano' se carga con las funciones
                        correspondientes

        :type compression: string
        :param compression: si es None se infiere la compresion segun 'nombreArchivo'
                            valores posibles 'zip', 'pgz' 'bzp2'

        url: http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
        # TODO
        """
        objeto = None
        if method != 'theano':
            from cupydle.dnn.utils import load
            objeto = load(nombreArchivo, compression)
        else:
            from cupydle.dnn.utils_theano import load
            objeto = load(nombreArchivo, compression)
        return objeto
    # END LOAD


if __name__ == "__main__":

    import os
    currentPath = os.getcwd()                               # directorio actual de ejecucion
    testPath    = currentPath + '/cupydle/test/mnist/'      # sobre el de ejecucion la ruta a los tests
    dataPath    = currentPath + '/cupydle/data/DB_mnist/'   # donde se almacenan la base de datos
    testFolder  = 'test0/'                                   # carpeta a crear para los tests
    fullPath    = testPath + testFolder

    if not os.path.exists(fullPath):        # si no existe la crea
        print('Creando la carpeta para el test en: ',fullPath)
        os.makedirs(fullPath)

    if not os.path.exists(dataPath):
        print("Creando la base de datos en:", dataPath)
        os.makedirs(dataPath)

    import subprocess
    subprocess.call(testPath + 'get_data.sh', shell=True)   # chequeo si necesito descargar los datos

    from cupydle.test.mnist.mnist import MNIST
    setName = "mnist"
    MNIST.prepare(dataPath, nombre=setName, compresion='bzip2')

    import argparse
    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre MNIST.')
    parser.add_argument('-g', '--guardar', action="store_true", dest="guardar", help="desea guardar (correr desde cero)", default=False)
    parser.add_argument('-m', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    guardar = args.guardar
    modelName = args.modelo
    #modelName = 'capa1.pgz'

    # para la prueba...
    from cupydle.test.mnist.mnist import MNIST
    from cupydle.test.mnist.mnist import open4disk
    from cupydle.test.mnist.mnist import save2disk

    # se leen de disco los datos
    mn = open4disk(filename=dataPath + setName, compression='bzip2')
    #mn.info                                        # muestra alguna informacion de la base de datos

    # obtengo todos los subconjuntos
    train_img,  train_labels= mn.get_training()
    test_img,   test_labels = mn.get_testing()
    val_img,    val_labels  = mn.get_validation()

    # parametros de la red
    n_visible = 784
    n_hidden  = 500
    batchSize = 20

    # creo la red
    red = RBM(n_visible=n_visible, n_hidden=n_hidden)

    red.setParams({'epsilonw':0.1})
    red.setParams({'epsilonvb':0.1})
    red.setParams({'epsilonhb':0.1})
    red.setParams({'initialmomentum':0.5})
    red.setParams({'weightcost':0.0002})
    red.setParams({'maxepoch':2})


    T = timer2()
    inicio = T.tic()

    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0:1], gibbsSteps=1)[0]
    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0], gibbsSteps=1)
    #MNIST.plot_one_digit((train_img/255.0).astype(numpy.float32)[0])
    #MNIST.plot_one_digit(salida)


    red.train(  data=(train_img/255.0).astype(numpy.float32),   # los datos los binarizo y convierto a float
                miniBatchSize=batchSize,
                pcd=True,
                gibbsSteps=1,
                validationData=(val_img/255.0).astype(numpy.float32),
                plotFilters=fullPath)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.elapsed(inicio, final)))

    # guardo los estadisticos
    #red.dibujarEstadisticos(show=True, save='estadisticos.png')
    red.dibujarEstadisticos(show=True, save=fullPath+'estadisticos.png')

    red.sampleo(data=(train_img/255.0).astype(numpy.float32),
                labels=train_labels)

    print('Guardando el modelo en ...', fullPath + modelName)
    inicio = T.tic()
    red.save(fullPath + modelName, absolutName=True)
    final = T.toc()
    print("Tiempo total para guardar: {}".format(T.elapsed(inicio, final)))

    red2 = RBM.load(fullPath + "coso.pgz")

    if numpy.allclose(red.w.get_value(), red2.w.get_value()):
        assert False
    else:
        print("no son iguales")

    print("FIN")
