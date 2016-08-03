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
        self.statistics = []
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
        dic = {}
        dic['errorTraining'] = 0.0
        dic['errorValidating'] = 0.0
        dic['errorTesting'] = 0.0
        dic['freeEnergy'] = 0.0

        self.statistics.append(dic)
        return 1

    def addStatistics(self, values):
        if not isinstance(values, dict):
            assert False, "necesito un diccionario"

        dic = {}
        dic['errorTraining'] = 0.0
        dic['errorValidating'] = 0.0
        dic['errorTesting'] = 0.0
        dic['freeEnergy'] = 0.0

        for key, val in values.items():
            if key in dic:
                dic[key] = values[key]
            else:
                assert False, "no exite el key" + str(key)

        self.statistics.append(dic)
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

    def markovChain_k(self, steps):
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
                                    name         = 'scan_oneStepCD2')

        # last step, dont sample, only necessary for parameters updates
        #linearSum2       = theano.tensor.dot(visibleActRec, self.w) + self.hidbiases
        #hiddenActRec    = self.activationFunction.activationProbablity(linearSum2)

        return ([visibleActRec, hiddenActData, probabilityP, probabilityN, linearSumP, linearSumN], updates)

    def markovChain_k2(self, steps):
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
                                    name         = 'scan_oneStepCD2')

        # last step, dont sample, only necessary for parameters updates
        linearSum2       = theano.tensor.dot(visibleActRec, self.w) + self.hidbiases
        hiddenActRec    = self.activationFunction.activationProbablity(linearSum2)

        return ([visibleActRec, hiddenActData, hiddenActRec, linearSumN], updates)



    def free_energy(self, vsample):
        ''' Function to compute the free energy '''
        wx_b = theano.tensor.dot(vsample, self.w) + self.hidbiases
        vbias_term = theano.tensor.dot(vsample, self.visbiases)
        hidden_term = theano.tensor.sum(theano.tensor.log(1 + theano.tensor.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def plotStatics(self, save=True, path=None):
        errorTrn = self.statistics['errorTraining']
        errorVal = self.statistics['errorValidation']
        freeEner = self.statistics['energyValidation']

        if errorTrn != []:
            fig1 = self.plot_error(errorTrn, tipo='training', show=False, save=save, path=path)
        if errorVal != []:
            fig2 = self.plot_error(errorVal, tipo='validation', show=False, save=save, path=path)
        if freeEner != []:
            fig3 = self.plot_error(freeEner, tipo='validation', show=False, save=save, path=path)

        #fig = plt.figure('Statics').clear()
        assert False
        """
        # Two subplots, the axes array is 1-d
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(x, y)
        axarr[0].set_title('Sharing X axis')
        axarr[1].scatter(x, y)
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
        """
        return

    def plot_filters(self, filename, path, binary=False):
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

        image.save(path + filename)

        return 1

    def get_cost_updates(self, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        def propup(vis):
            pre_sigmoid_activation = theano.tensor.dot(vis, self.w) + self.hidbiases
            return [pre_sigmoid_activation, theano.tensor.nnet.sigmoid(pre_sigmoid_activation)]

        def sample_h_given_v(v0_sample):
            pre_sigmoid_h1, h1_mean = propup(v0_sample)
            h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                                 n=1, p=h1_mean,
                                                 dtype=theanoFloat)
            return [pre_sigmoid_h1, h1_mean, h1_sample]

        def propdown(hid):
            pre_sigmoid_activation = theano.tensor.dot(hid, self.w.T) + self.visbiases
            return [pre_sigmoid_activation, theano.tensor.nnet.sigmoid(pre_sigmoid_activation)]

        def sample_v_given_h(h0_sample):
            pre_sigmoid_v1, v1_mean = propdown(h0_sample)
            v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theanoFloat)
            return [pre_sigmoid_v1, v1_mean, v1_sample]

        def gibbs_hvh(h0_sample):
            pre_sigmoid_v1, v1_mean, v1_sample = sample_v_given_h(h0_sample)
            pre_sigmoid_h1, h1_mean, h1_sample = sample_h_given_v(v1_sample)
            return [pre_sigmoid_v1, v1_mean, v1_sample,
                    pre_sigmoid_h1, h1_mean, h1_sample]

        pre_sigmoid_ph, ph_mean, ph_sample = sample_h_given_v(self.x)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = theano.tensor.mean(self.free_energy(self.x)) \
               - theano.tensor.mean(self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.internalParams):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * theano.tensor.cast(
                self.params['epsilonw'],
                dtype=theanoFloat
            )

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            """Stochastic approximation to the pseudo-likelihood"""

            # index of bit i in expression p(x_i | x_{\i})
            bit_i_idx = theano.shared(value=0, name='bit_i_idx')

            # binarize the input image by rounding to nearest integer
            xi = theano.tensor.round(self.x)

            # calculate free energy for the given bit configuration
            fe_xi = self.free_energy(xi)

            # flip bit x_i of matrix xi and preserve all other bits x_{\i}
            # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
            # the result to xi_flip, instead of working in place on xi.
            xi_flip = theano.tensor.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

            # calculate free energy with bit flipped
            fe_xi_flip = self.free_energy(xi_flip)

            # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
            cost = theano.tensor.mean(self.n_visible * theano.tensor.log(theano.tensor.nnet.sigmoid(fe_xi_flip -
                                                                fe_xi)))

            # increment bit_i_idx % number as part of updates
            updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

            monitoring_cost = cost
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = theano.tensor.mean(
                                theano.tensor.sum(
                                    self.x * theano.tensor.log(
                                        theano.tensor.nnet.sigmoid(pre_sigmoid_nvs[-1]))
                                            + (1 - self.x)
                                            * theano.tensor.log(1 - theano.tensor.nnet.sigmoid(pre_sigmoid_nvs[-1])),
                                axis=1
                                )
            )

        return monitoring_cost, updates

    def train(self, data, miniBatchSize=10, gibbsSteps=1, validationData=None, plotFilters=''):

        print("Entrenando una RBM, con [{}] unidades visibles y [{}] unidades ocultas".format(self.n_visible, self.n_hidden))
        print("Cantidad de ejemplos para el entrenamiento no supervisado: ", len(data))

        # convierto todos los datos a una variable shared de theano para llevarla a la GPU
        sharedData  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='TrainingData')

        # para la validacion
        if validationData is not None:
            sharedValidationData = theano.shared(numpy.asarray(a=validationData, dtype=theanoFloat), name='ValidationData')

        # Theano NODES.
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')
        steps = theano.tensor.lscalar('steps')

        # realizar la cadena de markov k veces
        (   [visibleActRec, hiddenActData, probabilityP, probabilityN, linearSumP, linearSumN],
            updates        ) = self.markovChain_k(steps)

        # la cadena finaliza con el ultimo muestreo de gibbs
        chain_end = visibleActRec[-1]

        # el costo de la red... diferencias de energia libre del inicio al final de la cadena
        cost = theano.tensor.mean(self.free_energy(self.x)) \
               - theano.tensor.mean(self.free_energy(chain_end))

        # NO se debe computar el gradiente a traves de las cadena de markov, solo la diferencia entre el primer y ultimo
        # para ello se deja constante la cadena... o variable mejor dicho
        #updates = self.buildUpdates(updates=updates, cost=cost, constant=chain_end)

        ###no se que carajo...
        assert False
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.internalParams):
            updates[param] = param - gparam * theano.tensor.cast(self.params['epsilonw'],
                            dtype=theanoFloat)

        monitoring_cost = theano.tensor.mean(
                                theano.tensor.sum(
                                    self.x * theano.tensor.log(
                                        theano.tensor.nnet.sigmoid(linearSumP[-1]))
                                            + (1 - self.x)
                                            * theano.tensor.log(1 - theano.tensor.nnet.sigmoid(linearSumP[-1])),
                                axis=1
                                )
        )

        # funcion princial
        trainer = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[monitoring_cost],
                        updates=updates,
                        givens={self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]}, ###
                        name='trainer'
        )


        if plotFilters is not None:
            # plot los filtros iniciales (sin entrenamiento)
            self.plot_filters(  filename='filtros_iniciales.pdf',
                                path=plotFilters)

        # cantidad de indices... para recorrer el set
        indexCount = int(data.shape[0]/miniBatchSize)
        mean_cost = numpy.Inf

        for epoch in range(0, self.params['maxepoch']):
            # imprimo algo de informacion sobre la terminal
            print(str('Starting Epoch {:>3d} '
                    + 'of {:>3d}, '
                    + 'errorTrn:{:>7.5f}, '
                    + 'errorVal:{:>7.5f}, '
                    + 'freeEnergy:{:>7.5f}').format(
                        epoch+1,
                        self.params['maxepoch'],
                        mean_cost,
                        0.0,
                        0.0),
                    end='\r')

            mean_cost = []
            for batch in range(0, indexCount):
                mean_cost += [trainer(batch, gibbsSteps)]

            mean_cost = numpy.mean(mean_cost)

            self.addStatistics({'errorTraining': mean_cost,
                                'errorValidating': 0.0,
                                'errorTesting': 0.0,
                                'freeEnergy': 0.0})

            if plotFilters is not None:
                self.plot_filters(filename='filters_at_epoch_{}.pdf'.format(epoch),
                                  path=plotFilters)

            # END SET
        # END epcoh
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior

        #print(self.statistics)
        return 1


    def train2(self, data, miniBatchSize=10, gibbsSteps=1, validationData=None, plotFilters=''):

        print("Entrenando una RBM, con [{}] unidades visibles y [{}] unidades ocultas".format(self.n_visible, self.n_hidden))
        print("Cantidad de ejemplos para el entrenamiento no supervisado: ", len(data))

        # convierto todos los datos a una variable shared de theano para llevarla a la GPU
        sharedData  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='TrainingData')

        # para la validacion
        if validationData is not None:
            sharedValidationData = theano.shared(numpy.asarray(a=validationData, dtype=theanoFloat), name='ValidationData')

        # Theano NODES.
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')
        """
        ##
        ##  de aca en adelante hasta el ##!!! puede borrarse
        ##
        # compute positive phase
        def oneStep_hvh(hsample):
            # positive
            # theano se da cuenta y no es necesario realizar un 'theano.tensor.tile' del
            # bias (dimension), ya que lo hace automaticamente
            linearSum = theano.tensor.dot(hsample, self.w) + self.hidbiases
            h1_mean   = self.activationFunction.deterministic(linearSum)
            h1_sample   = self.activationFunction.nonDeterminstic(linearSum)
            return [linearSum, h1_mean, linearSum]

        pre_sigmoid_ph, ph_mean, ph_sample = coso(self.x)

        chain_start = ph_sample
        ##!!!
        ##!!!
        ##!!!


        # realizar la cadena de markov k veces
        (   [visibleActRec,
             _,
             _,
             linearSum],
            updates        ) = self.markovChain_k(steps)

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = visibleActRec[-1]

        cost = theano.tensor.mean(self.free_energy(self.x)) - theano.tensor.mean(
            self.free_energy(chain_end))


        # build updates...
        updates = self.buildUpdates(updates=updates, cost=cost, constant=chain_end)


        cross_entropy = theano.tensor.mean(
                            theano.tensor.sum(
                                self.x
                                * theano.tensor.log(theano.tensor.nnet.sigmoid(linearSum[-1]))
                                + (1 - self.x)
                                * theano.tensor.log(1 - theano.tensor.nnet.sigmoid(linearSum[-1])),
                            axis=1
                            )
        )

        monitoring_cost = cross_entropy

        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train_rbm = theano.function(
            [miniBatchIndex, steps],
            monitoring_cost, # cost
            updates=updates,
            givens={
                self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
            },
            name='train_rbm'
        )
        """
        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((miniBatchSize, self.n_hidden),
                                                     dtype=theanoFloat),
                                         borrow=True)
        #cost, updates = self.get_cost_updates(persistent=None, k=steps)
        cost, updates = self.get_cost_updates(persistent=persistent_chain, k=steps)

        train_rbm = theano.function(
            [miniBatchIndex, steps],
            cost,
            updates=updates,
            givens={
                self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
            },
            name='train_rbm'
        )


        if plotFilters is not None:
            # plot los filtros iniciales (sin entrenamiento)
            self.plot_filters(  filename='filtros_iniciales.pdf',
                                path=plotFilters)

        # cantidad de indices... para recorrer el set
        indexCount = int(data.shape[0]/miniBatchSize)
        mean_cost = numpy.Inf

        for epoch in range(0, self.params['maxepoch']):
            # imprimo algo de informacion sobre la terminal
            print(str('Starting Epoch {:>3d} '
                    + 'of {:>3d}, '
                    + 'errorTrn:{:>7.5f}, '
                    + 'errorVal:{:>7.5f}, '
                    + 'freeEnergy:{:>7.5f}').format(
                        epoch+1,
                        self.params['maxepoch'],
                        mean_cost,
                        0.0,
                        0.0),
                    end='\r')

            mean_cost = []
            for batch in range(0, indexCount):
                mean_cost += [train_rbm(batch, gibbsSteps)]

            mean_cost = numpy.mean(mean_cost)

            self.addStatistics({'errorTraining': mean_cost,
                                'errorValidating': 0.0,
                                'errorTesting': 0.0,
                                'freeEnergy': 0.0})

            if plotFilters is not None:
                self.plot_filters(filename='filters_at_epoch_{}.pdf'.format(epoch),
                                  path=plotFilters)

            # END SET
        # END epcoh
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior

        #print(self.statistics)
        return 1

    def sampleo(self, data, labels=None, chains=20, samples=10, gibbsSteps=1000,
                patchesDim=(28,28), binary=False):
        """
        # TODO
        cambiar el gibbsSteps por una variable como en train...ver que pasa





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
            updates        ) = self.markovChain_k(gibbsSteps)

        # cambiar el valor de la cadenaFija al de la reconstruccion de las visibles
        updates.update({cadenaFija: visibleActRec[-1]})

        # funcion princial
        muestreo = theano.function(
                        inputs=[],
                        outputs=[probabilityN[-1], visibleActRec[-1]],
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
            probabilityN, _ = muestreo()

            print(' ... plotting sample {}'.format(idx))
            imageResults[(alto+1) * idx:(ancho+1) * idx + ancho, :] \
                = tile_raster_images(X=probabilityN,
                                    img_shape=(alto, ancho),
                                    tile_shape=(1, chains),
                                    tile_spacing=(1, 1)
                )

        # construct image
        image = Image.fromarray(imageResults)

        filename = "samples_" + str(labels[lista])
        filename = filename.replace(" ", "_")

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
            image2.save(filename + '_binary.pdf')

        image.save(filename + '.pdf')

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



    def save(self, filename=None, method='simple', compression=None, layerN=0, absolutName=False):
        """
        guarda a disco la instancia de la RBM
        method is simple, no guardo como theano
        :param filename:
        :param compression:
        :param layerN: numero de capa a la cual pertence la rbm (DBN)
        :param absolutName: si es true se omite los parametros a excepto el filename
        :return:
        """
        #from cupydle.dnn.utils import save as saver

        #if filename is not set
        if absolutName is False:
            if filename is None:
                filename = "V_" + str(self.n_visible) + "H_" + str(self.n_hidden) + "_" + time.strftime('%Y%m%d_%H%M')
            else:
                filename = filename + "_" + time.strftime('%Y-%m-%d_%H%M')

            if layerN != 0:
                filename = "L_" + str(layerN) + filename
        else:
            filename = filename

        if method != 'theano':
            from cupydle.dnn.utils import save as saver
            saver(objeto=self, filename=filename, compression=compression)
        else:
            with open(filename + '.zip','wb') as f:
                # arreglar esto
                from cupydle.dnn.utils_theano import save as saver
                saver(objeto=self, filename=filename, compression=compression)

        return
    # END SAVE

    @staticmethod
    def load(filename=None, method='simple', compression=None):
        """
        Carga desde archivo un objeto RBM guardado. Se distinguen los metodos
        de guardado, pickle simple o theano.

        :type filename: string
        :param filename: ruta completa al archivo donde se aloja la RBM

        :type method: string
        :param method: si es 'simple' se carga el objeto con los metodos estandar
                        para pickle, si es 'theano' se carga con las funciones
                        correspondientes

        :type compression: string
        :param compression: si es None se infiere la compresion segun 'filename'
                            valores posibles 'zip', 'pgz' 'bzp2'

        url: http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
        # TODO
        """
        objeto = None
        if method != 'theano':
            from cupydle.dnn.utils import load
            objeto = load(filename, compression)
        else:
            from cupydle.dnn.utils_theano import load
            objeto = load(filename, compression)
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
    red.setParams({'maxepoch':1})


    T = timer2()
    inicio = T.tic()

    red.train(  data=(train_img/255.0).astype(numpy.float32),   # los datos los binarizo y convierto a float
                miniBatchSize=batchSize,
                gibbsSteps=1,
                validationData=(val_img/255.0).astype(numpy.float32),
                plotFilters=fullPath)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.elapsed(inicio, final)))

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
