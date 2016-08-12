#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implementation of restricted Boltzmann machine on GP-GPU."""

# https://github.com/hunse/nef-rbm/blob/master/gaussian-binary-rbm.py
#

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
from cupydle.dnn.unidades import UnidadBinaria


"""




"""
from cupydle.dnn.utils import temporizador
try:
    import PIL.Image as Image
except ImportError:
    import Image

from cupydle.dnn.graficos import scale_to_unit_interval
from cupydle.dnn.graficos import tile_raster_images
from cupydle.dnn.graficos import filtrosConstructor

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
                theano_rng=None,    #seed para theano random
                ruta=''):
        """

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
            # inicializacion mejorarada de los pesos
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
            _hidbiases = numpy.zeros(shape=(self.n_hidden),
                                     dtype=theanoFloat)
            hidbiases = theano.shared(value=_hidbiases, name='hidbiases', borrow=True)
            del _hidbiases


        # create shared variable for visible units bias
        if visbiases is None:
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
        self.fnActivacionUnidEntrada = UnidadBinaria()
        self.fnActivacionUnidEntrada = None
        self.fnActivacionUnidSalida = UnidadBinaria()
        self.fnActivacionUnidSalida = None

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

        self.ruta = ruta
    # END INIT

    def _initParams(self):
        """ inicializa los parametros de la red, un diccionario"""
        self.params['epsilonw'] = 0.0
        self.params['epsilonvb'] = 0.0
        self.params['epsilonhb'] = 0.0
        self.params['weightcost'] = 0.0
        self.params['momentum'] = 0.0
        self.params['maxepoch'] = 0.0
        self.params['unidadesEntrada'] = UnidadBinaria()
        self.params['unidadesSalida'] = UnidadBinaria()
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


    def dibujarPesos(self, weight=None, save=None, path=None):
        """
        Grafica la matriz de pesos de (n_visible x n_ocultas) unidades

        :param weight: matriz de pesos asociada a una RBM, cualquiera
        """
        from cupydle.dnn.graficos import pesosConstructor
        assert False, "el plot son los histogramas"
        pesos = numpy.asarray(a=self.get_w())
        pesos = numpy.tile(A=pesos, reps=(20,1))
        print(self.get_w().shape, pesos.shape)
        pesosConstructor(pesos=pesos)
        return 1

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
            hiddenActData, probabilityP = self.fnActivacionUnidEntrada.activar(linearSumP)
            #negative
            linearSumN                  = theano.tensor.dot(hiddenActData, wG.T) + vbiasG
            visibleActRec, probabilityN = self.fnActivacionUnidSalida.activar(linearSumN)
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
        """
        Function to compute the free energy
        """
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

    def dibujarEstadisticos(self, mostrar=False, guardar=True):

        errorEntrenamiento  = self.estadisticos['errorEntrenamiento']
        errorValidacion     = self.estadisticos['errorValidacion']
        errorTesteo         = self.estadisticos['errorTesteo']
        mseEntrenamiento    = self.estadisticos['mseEntrenamiento']
        energiaLibreEntrenamiento = self.estadisticos['energiaLibreEntrenamiento']
        energiaLibreValidacion  = self.estadisticos['energiaLibreValidacion']

        f, axarr = matplotlib.pyplot.subplots(2, 3, sharex='col', sharey='row')

        axarr[0, 0] = self.crearDibujo(errorEntrenamiento, axarr[0, 0], titulo='Error de Entrenamiento', estilo='r-')
        axarr[0, 1] = self.crearDibujo(errorValidacion, axarr[0, 1], titulo='Error de Validacion', estilo='b-')
        axarr[0, 2] = self.crearDibujo(errorTesteo, axarr[0, 2], titulo='Error de Testeo')

        axarr[1, 0] = self.crearDibujo(mseEntrenamiento, axarr[1, 0], titulo='MSE de Entrenamiento')
        axarr[1, 1] = self.crearDibujo(energiaLibreEntrenamiento, axarr[1, 1], titulo='Energia Libre Entrenamiento')
        axarr[1, 2] = self.crearDibujo(energiaLibreValidacion, axarr[1, 2], titulo='Energia Libre Validacion')

        matplotlib.pyplot.tight_layout()

        if guardar:
            nombreArchivo= self.ruta + 'rbm_estadisticos.pdf'
            matplotlib.pyplot.savefig(nombreArchivo, bbox_inches='tight')

        if mostrar:
            matplotlib.pyplot.mostrar()
        return


    def dibujarFiltros(self, nombreArchivo='filtros.png', formaFiltro = (10,10), binary=False, mostrar=False):
        # mirar
        # http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
        # plot los filtros iniciales (sin entrenamiento)
        """
        cantidad = formaFiltro[0]*formaFiltro[1]
        ima = self.w.get_value(borrow=True).T[:cantidad]

        figura = filtrosConstructor(images=ima,
                                    titulo=nombreArchivo,
                                    formaFiltro=formaFiltro,
                                    nombreArchivo=self.ruta+nombreArchivo,
                                    mostrar=mostrar)
        """
        #figura.show()
        #"""
        # con el tile_raster...
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

        image.save(self.ruta + nombreArchivo)
        #"""

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
            visibleActRec, probabilityN = self.fnActivacionUnidEntrada.activar(linearSumN)
            # positive
            linearSumP                  = theano.tensor.dot(visibleActRec, wG) + hbiasG
            hiddenActData, probabilityP = self.fnActivacionUnidSalida.activar(linearSumP)
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

        # construyo las actualizaciones en los updates (variables shared)
        # segun el costo, constant es para proposito del gradiente
        updates=self.buildUpdates(updates=updates, cost=deltaEnergia, constant=chain_end)

        # como es una cadena persistente, la salida se debe actualizar, debido que es la entrada a la proxima

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

    def PersistentConstrastiveDivergence2(self, miniBatchSize, sharedData):

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
            visibleActRec, probabilityN = self.fnActivacionUnidEntrada.activar(linearSumN)
            # positive
            linearSumP                  = theano.tensor.dot(visibleActRec, wG) + hbiasG
            hiddenActData, probabilityP = self.fnActivacionUnidSalida.activar(linearSumP)
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
            hiddenActData, probabilityP = self.fnActivacionUnidEntrada.activar(linearSumP)
            #negative
            linearSumN                  = theano.tensor.dot(hiddenActData, wG.T) + vbiasG
            visibleActRec, probabilityN = self.fnActivacionUnidSalida.activar(linearSumN)
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

        # construyo las actualizaciones en los updates (variables shared)
        # segun el costo, constant es para proposito del gradiente
        updates=self.buildUpdates(updates=updates, cost=deltaEnergia, constant=chain_end)


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


    def buildUpdates(self, updates, cost, constant):
        """
        calcula las actualizaciones de la red sobre sus parametros w hb y vh
        con momento
        tasa de aprendizaje para los pesos, y los biases visibles y ocultos
        """
        # arreglo los parametros en caso de que no se hayan establecidos, considero las
        # tasa de aprendizaje para los bias igual al de los pesos
        if self.params['epsilonvb'] is None:
            self.params['epsilonvb'] = self.params['epsilonw']
        if self.params['epsilonvb'] == 0.0:
            self.params['epsilonvb'] = self.params['epsilonw']

        if self.params['epsilonhb'] is None:
            self.params['epsilonhb'] = self.params['epsilonw']
        if self.params['epsilonhb'] == 0.0:
            self.params['epsilonhb'] = self.params['epsilonw']
        assert self.params['epsilonw'] != 0.0, "La tasa de aprendizaje para los pesos no puede ser nula"


        # We must not compute the gradient through the gibbs sampling
        gparams = theano.tensor.grad(cost, self.internalParams, consider_constant=[constant])
        #gparams = theano.tensor.grad(cost, self.internalParams)

        # creo una lista con las actualizaciones viejas (shared guardadas)
        oldUpdates = [self.vishidinc, self.hidbiasinc, self.visbiasinc]

        # una lista de ternas [(w,dc/dw,w_old),(hb,dc/dhb,...),...]
        parametersTuples = zip(self.internalParams, gparams, oldUpdates)

        momentum = theano.tensor.cast(self.params['momentum'], dtype=theanoFloat)
        lr_pesos = theano.tensor.cast(self.params['epsilonw'], dtype=theanoFloat)
        lr_vbias = theano.tensor.cast(self.params['epsilonvb'], dtype=theanoFloat)
        lr_hbias = theano.tensor.cast(self.params['epsilonhb'], dtype=theanoFloat)

        otherUpdates = []

        for param, delta, oldUpdate in parametersTuples:
            # segun el paramerto tengo diferentes learning rates
            if param.name == self.w.name:
                paramUpdate = momentum * oldUpdate - lr_pesos * delta
            if param.name == self.visbiases.name:
                paramUpdate = momentum * oldUpdate - lr_vbias * delta
            if param.name == self.hidbiases.name:
                paramUpdate = momentum * oldUpdate - lr_hbias * delta
            #param es la variable o paramatro
            #paramUpdate son los incrementos
            #newParam es la variable mas su incremento
            newParam = param + paramUpdate
            otherUpdates.append((param, newParam)) # w-> w+inc ...
            otherUpdates.append((oldUpdate, paramUpdate))  #winc_old -> winc_new

        updates.update(otherUpdates)
        return updates

    def train(self, data, miniBatchSize=10, pcd=True, gibbsSteps=1, validationData=None, filtros=False, printCompacto=False):
        # mirar:
        # https://github.com/hunse/nef-rbm/blob/master/gaussian-binary-rbm.py
        #
        #
        #
        # ahi esta la actualziacion como lo hace hinton

        print("Entrenando una RBM, con [{}] unidades visibles y [{}] unidades ocultas".format(self.n_visible, self.n_hidden))
        print("Cantidad de ejemplos para el entrenamiento no supervisado: ", len(data))

        # convierto todos los datos a una variable shared de theano para llevarla a la GPU
        sharedData  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='TrainingData')

        # para la validacion
        if validationData is not None:
            sharedValidationData = theano.shared(numpy.asarray(a=validationData, dtype=theanoFloat), name='ValidationData')

        trainer = None
        self.fnActivacionUnidEntrada = self.params['unidadesEntrada']
        self.fnActivacionUnidSalida = self.params['unidadesSalida']
        if pcd:
            print("Entrenando con Divergencia Contrastiva Persistente, {} pasos de Gibss.".format(gibbsSteps))
            trainer = self.PersistentConstrastiveDivergence(miniBatchSize, sharedData)
        else:
            print("Entrenando con Divergencia Contrastiva, {} pasos de Gibss.".format(gibbsSteps))
            trainer = self.ConstrastiveDivergence(miniBatchSize, sharedData)

        if filtros:
            # plot los filtros iniciales (sin entrenamiento)
            self.dibujarFiltros(nombreArchivo='filtros_epoca_0.pdf')



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

            if filtros:
                self.dibujarFiltros(nombreArchivo='filtros_epoca_{}.pdf'.format(epoch+1))

            # END SET
        # END epoch
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior

        self.dibujarEstadisticos()
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

    def guardar(self, nombreArchivo=None, method='simple', compression=None):
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

        ruta = self.ruta
        import time

        if nombreArchivo is None:
            nombreArchivo = ruta + "RBM_V" + str(self.n_visible) + "H" + str(self.n_hidden) + "_" + time.strftime('%Y%m%d_%H%M') + '.zip'
        else:
            nombreArchivo = ruta + nombreArchivo

        if method != 'theano':
            from cupydle.dnn.utils import save as saver
            saver(objeto=self, filename=nombreArchivo, compression=compression)
        else:
            with open(nombreArchivo,'wb') as f:
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
    assert False, str(__file__ + " No es un modulo")
