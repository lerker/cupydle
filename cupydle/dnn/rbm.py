#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implementacion de Maquina de Boltzmann Restringidas en GP-GPU"""

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

from cupydle.dnn.unidades import UnidadBinaria
from cupydle.dnn.loss import errorCuadraticoMedio

from cupydle.dnn.utils_theano import gpu_info
from cupydle.dnn.utils_theano import calcular_chunk
from cupydle.dnn.utils_theano import calcular_memoria_requerida

from warnings import warn


"""




"""
from cupydle.dnn.utils import temporizador
try:
    import PIL.Image as Image
except ImportError:
    import Image


from cupydle.dnn.graficos import imagenTiles

import matplotlib.pyplot
import math

class RBM(object):
    """Restricted Boltzmann Machine on GP-GPU (RBM-GPU)  """

    verbose=True

    def __init__(self,
                 n_visible=784,
                 n_hidden=500,
                 w=None,
                 visbiases=None,
                 hidbiases=None,
                 numpy_rng=None,
                 theano_rng=None,
                 nombre=None,
                 ruta=''):
        """
        :type n_visible: int
        :param n_visible: cantidad de neuronas visibles
        :type n_hidden: int
        :param n_hidden: cantidad de neuronas ocultas
        :type w: shared variable theano, numpy.ndarray. size=(n_visible, n_hidden)
        :param w: matriz de pesos
        :type visbiases: shared variable theano, numpy.ndarray. size=(n_visible)
        :param visbiases: vector de bias visible
        :type hidbiases: shared variable theano, numpy.ndarray. size=(n_hidden)
        :param visbiases: vector de bias oculto
        :type numpy_rnd: int
        :param numpy_rnd: semilla para la generacion de numeros aleatorios numpy
        :type theano_rnd: int
        :param theano_rnd: semilla para la generacion de numeros aleatorios theano
        :type ruta: string
        :param ruta: ruta de almacenamiento de los datos generados
        """

        self.n_visible = n_visible
        self.n_hidden  = n_hidden

        # create a number generator (fixed) for test NUMPY
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        # create a number generator (fixed) for test THEANO
        if theano_rng is None:
            #theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            theano_rng = RandomStreams(seed=1234)

        if w is None:
            w = self._initW(numpy_rng=numpy_rng, metodo="mejorado")

        if visbiases is None:
            visbiases = self._initBiasVisible()

        if hidbiases is None:
            hidbiases = self._initBiasOculto()

        # w es una matriz de numpy pasada a la funcion?
        if w is not None and isinstance(w, numpy.ndarray):
            w = theano.shared(value=w, name='w', borrow=True)

        ########
        self.visbiases  = visbiases
        self.hidbiases  = hidbiases
        self.w          = w
        self.numpy_rng  = numpy_rng
        self.theano_rng = theano_rng

        # funcion de activacion sigmodea para cada neurona (porbabilidad->binomial)
        self.fnActivacionUnidEntrada = None
        self.fnActivacionUnidSalida  = None

        # buffers para el almacenamiento temporal de las variables
        # momento-> incrementos, historicos
        self.vishidinc   = theano.shared(value=numpy.zeros(shape=(self.n_visible, self.n_hidden), dtype=theanoFloat), name='vishidinc')
        self.hidbiasinc  = theano.shared(value=numpy.zeros(shape=(self.n_hidden), dtype=theanoFloat), name='hidbiasinc')
        self.visbiasinc  = theano.shared(value=numpy.zeros(shape=(self.n_visible), dtype=theanoFloat), name='visbiasinc')

        # para las derivadas, parametro que componene a la red
        self.internalParams = [self.w, self.hidbiases, self.visbiases]

        # seguimiento del grafo theano, root
        self.x = theano.tensor.matrix(name="x")

        # parametros para el entrenamiento
        self.params = {}
        self._initParams()

        # estadisticos
        self.estadisticos = {}
        self._initStatistics()

        self.nombre = 'rbm' if nombre is None else nombre

        self.ruta = ruta
    # END INIT

    def _initW(self, numpy_rng, metodo='mejorado'):
        """
        Inicializacion mejorarada de los pesos

        :type numpy_rng: int
        :param numpy_rng: semilla para la generacion random
        :type metodo: string
        :param metodo: seleccion del metodo de inicializacion (comun, mejorado)
        """
        if metodo=='comun':
            _w = numpy.asarray(
                numpy.random.normal(
                    0, 0.01, (self.n_visible, self.n_hidden)), dtype=theanoFloat)
        elif metodo=='mejorado':
            _w = numpy.asarray(
                numpy_rng.uniform(
                    low= -4 * numpy.sqrt(6. / (self.n_visible + self.n_hidden)),
                    high= 4 * numpy.sqrt(6. / (self.n_visible + self.n_hidden)),
                    size=(self.n_visible, self.n_hidden)),
                dtype=theanoFloat)
        else:
            raise Exception("Metodo de inicializacion de los pesos desconocido: "
                     + str(metodo))

        # se pasa el buffer al namespace de la GPU
        w = theano.shared(value=_w, name='w', borrow=True)

        # se libera memoria de la CPU
        del _w

        return w

    def _initBiasVisible(self):
        _visbiases = numpy.zeros(shape=(self.n_visible), dtype=theanoFloat)
        visbiases = theano.shared(value=_visbiases, name='visbiases', borrow=True)
        del _visbiases

        return visbiases

    def _initBiasOculto(self):
        _hidbiases = numpy.zeros(shape=(self.n_hidden), dtype=theanoFloat)
        hidbiases = theano.shared(value=_hidbiases, name='hidbiases', borrow=True)
        del _hidbiases

        return hidbiases

    def _initParams(self):
        """ inicializa los parametros de la red, un diccionario"""
        self.params['epsilonw'] = 0.0
        self.params['epsilonvb'] = 0.0
        self.params['epsilonhb'] = 0.0
        self.params['weightcost'] = 0.0
        self.params['momentum'] = 0.0
        self.params['epocas'] = 0.0
        self.params['unidadesVisibles'] = UnidadBinaria()
        self.params['unidadesOcultas'] = UnidadBinaria()
        self.params['dropoutVisibles'] = 1.0
        self.params['dropoutOcultas'] = 1.0

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

    @property
    def get_w(self):
        return self.w.get_value(borrow=True)

    @property
    def get_biasVisible(self):
        return self.visbiases.get_value(borrow=True)

    @property
    def get_biasOculto(self):
        return self.hidbiases.get_value(borrow=True)

    @property
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

    def energiaLibre(self, vsample):
        """
        Calcula la energia libre F(v) = - log sum_h exp(-E(v,h)).

        Parametros
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Retornos
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        wx_b = theano.tensor.dot(vsample, self.w) + self.hidbiases
        vbias_term = theano.tensor.dot(vsample, self.visbiases)
        hidden_term = theano.tensor.sum(theano.tensor.log(1 + theano.tensor.exp(wx_b)), axis=1)
        #return -hidden_term - vbias_term +  0.5* theano.tensor.sum(vbias_term**2, axis=0)
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


    def dibujarFiltros(self, nombreArchivo='filtros.png', automatico=True, formaFiltro = (10,10), binary=False, mostrar=False):

        assert isinstance(formaFiltro, tuple), "Forma filtro debe ser una tupla (X,Y)"

        # corrige si lo indico... sino es porque quiero asi, en modo auto es cuadrado
        if automatico:
            cantPesos = self.w.get_value(borrow=True).T.shape[1]

            # ancho/alto tile es la cantidad dividido su raiz y tomando el floor
            # a entero
            anchoTile= int(cantPesos // math.sqrt(cantPesos))
            cantPesosCorregido = anchoTile ** 2
            if cantPesos != cantPesosCorregido:
                cantPesos = cantPesosCorregido

            formaImagen = (anchoTile, anchoTile)


        image = Image.fromarray(
            imagenTiles(
                # se mandan las imagenes desde 0 a hasta donde cuadre
                X=self.w.get_value(borrow=True).T[:,0:cantPesos],
                img_shape=formaImagen,
                tile_shape=formaFiltro,
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

        # si le pase un nombre es porque lo quiero guardar
        if nombreArchivo is not None:
            image.save(self.ruta + nombreArchivo)
        else:
            nombreArchivo=''

        if mostrar:
            image.show(title=nombreArchivo)

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
        #TODO esto estaba, pero arroja un error (paraque si no lo uso)?
        #updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        # TODO es necesario retornar las updates?
        # ni siquiera las utiliza en el algoritmo.. medio inutil pasarla o retornarla
        return cost

    def pseudoLikelihoodCost_bengio(self, updates=None):
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
        #TODO esto estaba, pero arroja un error (paraque si no lo uso)?
        #updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        # TODO es necesario retornar las updates?
        # ni siquiera las utiliza en el algoritmo.. medio inutil pasarla o retornarla
        return cost

    def muestrearVdadoH(self, muestra_H0):
        salidaLineal_V1 = theano.tensor.dot(muestra_H0, self.w.T) + self.visbiases
        muestra_V1, probabilidad_V1 = self.fnActivacionUnidEntrada.activar(salidaLineal_V1)

        return [salidaLineal_V1, probabilidad_V1, muestra_V1]

    def muestrearVdadoH_dropout(self, muestra_H0, mask):
        salidaLineal_V1 = theano.tensor.dot(muestra_H0, self.w.T) + self.visbiases
        #muestra_V1, probabilidad_V1 = self.fnActivacionUnidEntrada.activar(salidaLineal_V1*theano.tensor.cast(mask, theanoFloat))

        salidaLineal_V1*=theano.tensor.cast(mask, theanoFloat)

        muestra_V1, probabilidad_V1 = self.fnActivacionUnidEntrada.activar(salidaLineal_V1)


        #muestra_V1 *= theano.tensor.cast(mask, theanoFloat)
        #probabilidad_V1*=theano.tensor.cast(mask, theanoFloat)

        return [salidaLineal_V1, probabilidad_V1, muestra_V1]


    def muestrearHdadoV_dropout(self, muestra_V0, mask):
        # segun hinton
        # srivastava2014dropout.pdf
        # Dropout: A Simple Way to Prevent Neural Networks from Overfitting



        #theano.tensor.cast(mask, theanoFloat)
        salidaLineal_H1 = theano.tensor.dot(muestra_V0, self.w) + self.hidbiases
        #muestra_H1, probabilidad_H1 = self.fnActivacionUnidSalida.activar(salidaLineal_H1*theano.tensor.cast(mask, theanoFloat))

        salidaLineal_H1 *= theano.tensor.cast(mask, theanoFloat)

        muestra_H1, probabilidad_H1 = self.fnActivacionUnidSalida.activar(salidaLineal_H1)


        #muestra_H1*=theano.tensor.cast(mask, theanoFloat)
        #probabilidad_H1*=theano.tensor.cast(mask, theanoFloat)

        return [salidaLineal_H1, probabilidad_H1, muestra_H1]


    def muestrearHdadoV(self, muestra_V0):
        salidaLineal_H1 = theano.tensor.dot(muestra_V0, self.w) + self.hidbiases
        muestra_H1, probabilidad_H1 = self.fnActivacionUnidSalida.activar(salidaLineal_H1)

        return [salidaLineal_H1, probabilidad_H1, muestra_H1]

    def gibbsHVH(self, muestra, steps):
        # un paso de CD es Hidden->Visible->Hidden
        def unPaso(muestraH, w, vbias, hbias):

            salidaLineal_V1, probabilidad_V1, muestra_V1 = self.muestrearVdadoH(muestraH)
            salidaLineal_H1, probabilidad_H1, muestra_H1 = self.muestrearHdadoV(muestra_V1)

            return [salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1]

        ( [salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_unPasoGibbsHVH')
        return ([salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1], updates)


    def gibbsVHV(self, muestra, steps):
        # un paso de CD es Visible->Hidden->Visible
        def unPaso(muestraV, w, vbias, hbias):

            salidaLineal_H1, probabilidad_H1, muestra_H1 = self.muestrearHdadoV(muestraV)
            salidaLineal_V1, probabilidad_V1, muestra_V1 = self.muestrearVdadoH(muestra_H1)

            return [salidaLineal_H1, probabilidad_H1, muestra_H1, salidaLineal_V1, probabilidad_V1, muestra_V1]

        ( [salidaLineal_H1, probabilidad_H1, muestra_H1, salidaLineal_V1, probabilidad_V1, muestra_V1],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_unPasoGibbsVHV')

        return ([salidaLineal_H1, probabilidad_H1, muestra_H1, salidaLineal_V1, probabilidad_V1, muestra_V1], updates)


    def pasoGibbsPersistente(self, muestra, steps):
        # si fuera de un solo paso, se samplea las ocultas, re recupera las visibles
        # y vuelve a samplear las ocultas (aunque no se utilece el samble ultimo, solo la probabilidad)

        # si fuera de un solo paso, se samplea las ocultas, re recupera las visibles
        # y vuelve a samplear las ocultas (aunque no se utilece el samble ultimo, solo la probabilidad)

        # un paso de CD es Visible->Hidden->Visible
        def unPaso(muestraV, w, vbias, hbias):

            salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV(muestraV)
            salidaLineal_Vk, probabilidad_Vk, muestra_Vk = self.muestrearVdadoH(muestra_H0)

            return [salidaLineal_H0, probabilidad_H0, muestra_H0, salidaLineal_Vk, probabilidad_Vk, muestra_Vk]

        ( [salidaLineal_H0, probabilidad_H0, muestra_H0, salidaLineal_Vk, probabilidad_Vk, muestra_Vk],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_pasosGibbsPersistente')


        return ([salidaLineal_H0[-1], probabilidad_H0[-1], muestra_H0[-1], salidaLineal_Vk[-1], probabilidad_Vk[-1], muestra_Vk[-1]], updates)

    def DivergenciaContrastivaPersistente(self, miniBatchSize, sharedData):
        # con dropout
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

                # dropout
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        from numpy.random import randint as npRandint
        theanoGenerator = RandomStreams(seed=npRandint(1, 1000))

        # tamanio de la mascara

        visibleDropout=self.params['dropoutVisibles']
        hiddenDropout=self.params['dropoutOcultas']
        dropoutMaskVisible = theanoGenerator.binomial(
                                size=self.hidbiases.shape,
                                n=1, p=visibleDropout,
                                dtype=theanoFloat)
        dropoutMaskVisible2 = theanoGenerator.binomial(
                                size=self.x.shape,
                                n=1, p=visibleDropout,
                                dtype=theanoFloat)
        droppedOutVisible = self.x * dropoutMaskVisible2

        dropoutMaskHidden = theanoGenerator.binomial(
                                size=self.visbiases.shape,
                                n=1, p=hiddenDropout,
                                dtype=theanoFloat)


        # primer termino del grafiente de la funcion de verosimilitud
        # la esperanza sobre el conjunto del dato
        salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV_dropout(droppedOutVisible, mask=dropoutMaskVisible)

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((miniBatchSize, self.n_hidden),
                                                     dtype=theanoFloat),
                                         borrow=True)

        # aproximo el segundo termino del gradiente de la funcion de verosimilitud
        # por medio de una cadena de Gibbs
        ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_Hk, probabilidad_Hk, muestra_Hk],
         updates) = self.pasoGibbs22(persistent_chain, steps, drop1=dropoutMaskVisible, drop2=dropoutMaskHidden)


        energia_dato = theano.tensor.mean(self.energiaLibre(droppedOutVisible))
        # TODO ver si le meto la salida lineal o bien la probabilidad
        energia_modelo = theano.tensor.mean(self.energiaLibre(muestra_Vk))
        #energia2 = theano.tensor.mean(self.energiaLibre(salidaLineal_Vk))

        costo1 = energia_dato - energia_modelo

        # construyo las actualizaciones en los updates (variables shared)
        updates=self.construirActualizaciones(
                    updatesOriginal=updates,
                    probabilidad_H0=probabilidad_H0,
                    probabilidad_Vk=probabilidad_Vk,
                    probabilidad_Hk=probabilidad_Hk,
                    muestra_Vk=muestra_Vk,
                    miniBatchSize=miniBatchSize)

        costo2 = self.reconstructionCost(salidaLineal_Vk)

        costo3 = errorCuadraticoMedio(droppedOutVisible, probabilidad_Vk)

        #updates[persistent_chain] = muestra_H1[-1]
        updates.append((persistent_chain, muestra_Hk))


        train_rbm = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[costo2, costo3, costo1],
                        updates=updates,
                        givens={
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_pcd'
        )

        return train_rbm

    def DivergenciaContrastivaPersistente_backup(self, miniBatchSize, sharedData):
        # sin dropout
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')


        # primer termino del grafiente de la funcion de verosimilitud
        # la esperanza sobre el conjunto del dato
        salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV(self.x)

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((miniBatchSize, self.n_hidden),
                                                     dtype=theanoFloat),
                                         borrow=True)

        # aproximo el segundo termino del gradiente de la funcion de verosimilitud
        # por medio de una cadena de Gibbs
        ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_Hk, probabilidad_Hk, muestra_Hk],
         updates) = self.pasoGibbs2(persistent_chain, steps)


        energia_dato = theano.tensor.mean(self.energiaLibre(self.x))
        # TODO ver si le meto la salida lineal o bien la probabilidad
        energia_modelo = theano.tensor.mean(self.energiaLibre(muestra_Vk))
        #energia2 = theano.tensor.mean(self.energiaLibre(salidaLineal_Vk))

        costo1 = energia_dato - energia_modelo

        # construyo las actualizaciones en los updates (variables shared)
        updates=self.construirActualizaciones(
                    updatesOriginal=updates,
                    probabilidad_H0=probabilidad_H0,
                    probabilidad_Vk=probabilidad_Vk,
                    probabilidad_Hk=probabilidad_Hk,
                    muestra_Vk=muestra_Vk,
                    miniBatchSize=miniBatchSize)

        costo2 = self.reconstructionCost(salidaLineal_Vk)

        costo3 = self.reconstructionCost_MSE(probabilidad_Vk)

        #updates[persistent_chain] = muestra_H1[-1]
        updates.append((persistent_chain, muestra_Hk))


        train_rbm = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[costo2, costo3, costo1],
                        updates=updates,
                        givens={
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_pcd'
        )

        return train_rbm


    def DivergenciaContrastivaPersistente_bengio(self, miniBatchSize, sharedData):

        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((miniBatchSize, self.n_hidden),
                                                     dtype=theanoFloat),
                                         borrow=True)


        ([ _, _, muestra_V1, _, _, muestra_H1],
          updates ) = self.gibbsHVH(persistent_chain, steps)

        chain_end = muestra_V1[-1]

        cost = theano.tensor.mean(self.energiaLibre(self.x)) - theano.tensor.mean(
            self.energiaLibre(chain_end))

        deltaEnergia = cost

        # construyo las actualizaciones en los updates (variables shared)
        # segun el costo, constant es para proposito del gradiente
        updates=self.buildUpdates(updates=updates, cost=deltaEnergia, constant=chain_end)

        # como es una cadena persistente, la salida se debe actualizar, debido que es la entrada a la proxima

        updates[persistent_chain] = muestra_H1[-1]

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
        warn("No se deberia utilizar mas, ver <<loss>>")
        # mean squared error, error cuadratico medio esta mal este de aca

        mse = theano.tensor.mean(
                theano.tensor.sum(
                    theano.tensor.sqr( self.x - reconstrucciones ), axis=1
                )
        )

        return mse

    def pasoGibbs(self, muestra, steps):
        # si fuera de un solo paso, se samplea las ocultas, re recupera las visibles
        # y vuelve a samplear las ocultas (aunque no se utilece el samble ultimo, solo la probabilidad)

        # un paso de CD es Visible->Hidden->Visible
        def unPaso(muestraV, w, vbias, hbias):

            salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV(muestraV)
            salidaLineal_Vk, probabilidad_Vk, muestra_Vk = self.muestrearVdadoH(muestra_H0)

            return [salidaLineal_H0, probabilidad_H0, muestra_H0, salidaLineal_Vk, probabilidad_Vk, muestra_Vk]

        ( [salidaLineal_H0, probabilidad_H0, muestra_H0, salidaLineal_Vk, probabilidad_Vk, muestra_Vk],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_pasosGibbs')

        salidaLineal_H0 = salidaLineal_H0[-1]
        probabilidad_H0 = probabilidad_H0[-1]
        muestra_H0 = muestra_H0[-1]
        salidaLineal_Vk = salidaLineal_Vk[-1]
        probabilidad_Vk = probabilidad_Vk[-1]
        muestra_Vk = muestra_Vk[-1]

        # ultimo paso
        salidaLineal_HK, probabilidad_HK, muestra_HK = self.muestrearHdadoV(muestra_Vk)

        return ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_HK, probabilidad_HK, muestra_HK], updates)

    #h,v,h
    def pasoGibbs2(self, muestra, steps):
        def unPaso(muestraH, w, vbias, hbias):
            salidaLineal_V1, probabilidad_V1, muestra_V1 = self.muestrearVdadoH(muestraH)
            salidaLineal_H1, probabilidad_H1, muestra_H1 = self.muestrearHdadoV(muestra_V1)

            return [salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1]

        ( [salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_pasoGibbs2')
        return ([salidaLineal_V1[-1], probabilidad_V1[-1], muestra_V1[-1], salidaLineal_H1[-1], probabilidad_H1[-1], muestra_H1[-1]], updates)


    def pasoGibbs22(self, muestra, steps, drop1, drop2):
        def unPaso(muestraH, w, vbias, hbias, drop11, drop22):
            salidaLineal_V1, probabilidad_V1, muestra_V1 = self.muestrearVdadoH_dropout(muestraH, drop22)
            #salidaLineal_V1, probabilidad_V1, muestra_V1 = self.muestrearVdadoH(muestraH)
            salidaLineal_H1, probabilidad_H1, muestra_H1 = self.muestrearHdadoV_dropout(muestra_V1, drop11)

            return [salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1]

        ( [salidaLineal_V1, probabilidad_V1, muestra_V1, salidaLineal_H1, probabilidad_H1, muestra_H1],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases, drop1, drop2],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_pasoGibbs2')
        return ([salidaLineal_V1[-1], probabilidad_V1[-1], muestra_V1[-1], salidaLineal_H1[-1], probabilidad_H1[-1], muestra_H1[-1]], updates)


    def DivergenciaContrastiva(self, miniBatchSize, sharedData, visibleDropout=1):
        ## con dropout
        ##
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # dropout
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        from numpy.random import randint as npRandint
        theanoGenerator = RandomStreams(seed=npRandint(1, 1000))

        # tamanio de la mascara

        # corregir esto como lo hizo mrosca
        sized = (miniBatchSize, sharedData.get_value(borrow=True).shape[1])

        visibleDropout=self.params['dropoutVisibles']
        hiddenDropout=self.params['dropoutOcultas']
        dropoutMaskVisible = theanoGenerator.binomial(
                                size=self.hidbiases.shape,
                                n=1, p=visibleDropout,
                                dtype=theanoFloat)
        dropoutMaskVisible2 = theanoGenerator.binomial(
                                size=self.x.shape,
                                n=1, p=visibleDropout,
                                dtype=theanoFloat)
        droppedOutVisible = self.x * dropoutMaskVisible2

        dropoutMaskHidden = theanoGenerator.binomial(
                                size=self.visbiases.shape,
                                n=1, p=hiddenDropout,
                                dtype=theanoFloat)



        # primer termino del grafiente de la funcion de verosimilitud
        # la esperanza sobre el conjunto del dato
        salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV_dropout(droppedOutVisible, mask=dropoutMaskVisible)
        #salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV_dropout(droppedOutVisible, mask=dropoutMaskVisible, p=visibleDropout)

        # aproximo el segundo termino del gradiente de la funcion de verosimilitud
        # por medio de una cadena de Gibbs
        ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_Hk, probabilidad_Hk, muestra_Hk],
         updates) = self.pasoGibbs22(muestra_H0, steps, drop1=dropoutMaskVisible, drop2=dropoutMaskHidden)


        #energia_dato = theano.tensor.mean(self.energiaLibre(self.x))
        energia_dato = theano.tensor.mean(self.energiaLibre(droppedOutVisible))
        # TODO ver si le meto la salida lineal o bien la probabilidad
        energia_modelo = theano.tensor.mean(self.energiaLibre(probabilidad_Vk))
        #energia2 = theano.tensor.mean(self.energiaLibre(salidaLineal_Vk))

        costo1 = energia_dato - energia_modelo

        # construyo las actualizaciones en los updates (variables shared)
        updates=self.construirActualizaciones(
                    updatesOriginal=updates,
                    probabilidad_H0=probabilidad_H0,
                    probabilidad_Vk=probabilidad_Vk,
                    probabilidad_Hk=probabilidad_Hk,
                    muestra_Vk=muestra_Vk,
                    miniBatchSize=miniBatchSize)

        costo2 = self.reconstructionCost(salidaLineal_Vk)


        #costo3 = self.reconstructionCost_MSE(droppedOutVisible, probabilidad_Vk)
        costo3 = errorCuadraticoMedio(droppedOutVisible, probabilidad_Vk)

        """
        estadisticos_update = []
        estadistico = (self.sharedEstadisticos, self.sharedEstadisticos + [costo2, costo3, costo1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        estadisticos_update.append(estadistico)
        updates += estadisticos_update
        """

        train_rbm = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[costo2, costo3, costo1],
                        updates=updates,
                        givens={
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_cd'
        )

        return train_rbm


    def DivergenciaContrastiva_Backup(self, miniBatchSize, sharedData):
        # sin dropout

        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')


        # primer termino del grafiente de la funcion de verosimilitud
        # la esperanza sobre el conjunto del dato
        salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV(self.x)

        # aproximo el segundo termino del gradiente de la funcion de verosimilitud
        # por medio de una cadena de Gibbs
        ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_Hk, probabilidad_Hk, muestra_Hk],
         updates) = self.pasoGibbs2(muestra_H0, steps)


        energia_dato = theano.tensor.mean(self.energiaLibre(self.x))
        # TODO ver si le meto la salida lineal o bien la probabilidad
        energia_modelo = theano.tensor.mean(self.energiaLibre(probabilidad_Vk))
        #energia2 = theano.tensor.mean(self.energiaLibre(salidaLineal_Vk))

        costo1 = energia_dato - energia_modelo

        # construyo las actualizaciones en los updates (variables shared)
        updates=self.construirActualizaciones(
                    updatesOriginal=updates,
                    probabilidad_H0=probabilidad_H0,
                    probabilidad_Vk=probabilidad_Vk,
                    probabilidad_Hk=probabilidad_Hk,
                    muestra_Vk=muestra_Vk,
                    miniBatchSize=miniBatchSize)

        costo2 = self.reconstructionCost(salidaLineal_Vk)

        costo3 = self.reconstructionCost_MSE(probabilidad_Vk)

        train_rbm = theano.function(
                        inputs=[miniBatchIndex, steps],
                        outputs=[costo2, costo3, costo1],
                        updates=updates,
                        givens={
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_cd'
        )

        return train_rbm

    def DivergenciaContrastiva_begio(self, miniBatchSize, sharedData):
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # Visible -> Hidden -> Visible
        ([ _, _, _, salidaLineal_V1, _, muestra_V1],
          updates) = self.gibbsVHV(self.x, steps)

        chain_end = muestra_V1[-1]

        cost = theano.tensor.mean(self.energiaLibre(self.x)) - theano.tensor.mean(
            self.energiaLibre(chain_end))

        deltaEnergia = cost

        # construyo las actualizaciones en los updates (variables shared)
        # segun el costo, constant es para proposito del gradiente
        updates=self.buildUpdates(updates=updates, cost=deltaEnergia, constant=chain_end)


        monitoring_cost = self.reconstructionCost(salidaLineal_V1[-1])

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

    def construirActualizaciones(self, updatesOriginal, probabilidad_H0, probabilidad_Vk, probabilidad_Hk, muestra_Vk, miniBatchSize):
        """
        calcula las actualizaciones de la red sobre sus parametros w hb y vh
        con momento
        tasa de aprendizaje para los pesos, y los biases visibles y ocultos
        """
        # arreglo los parametros en caso de que no se hayan establecidos, considero las
        # tasa de aprendizaje para los bias igual al de los pesos
        if self.params['epsilonvb'] is None:
            self.params['epsilonvb'] = self.params['epsilonw']

        if numpy.any(self.params['epsilonvb'] == 0.0):
            self.params['epsilonvb'] = self.params['epsilonw']

        if self.params['epsilonhb'] is None:
            self.params['epsilonhb'] = self.params['epsilonw']
        if numpy.any(self.params['epsilonhb'] == 0.0):
            self.params['epsilonhb'] = self.params['epsilonw']
        assert numpy.any(self.params['epsilonw'] != 0.0), "La tasa de aprendizaje para los pesos no puede ser nula"

        momentum = theano.tensor.cast(self.params['momentum'], dtype=theanoFloat)
        lr_pesos = theano.tensor.cast(self.params['epsilonw'], dtype=theanoFloat)
        lr_vbias = theano.tensor.cast(self.params['epsilonvb'], dtype=theanoFloat)
        lr_hbias = theano.tensor.cast(self.params['epsilonhb'], dtype=theanoFloat)
        # el escalado del dropout es realizado en el ajuste fino... no aca
        """
        dropout = self.params['dropoutVisibles']
        if dropout in [0.0, 0]:
            dropout = 0.00005
        dropout = theano.tensor.cast(dropout, dtype=theanoFloat)
        """
        updates = []

        # actualizacion de W
        positiveDifference = theano.tensor.dot(self.x.T, probabilidad_H0)
        negativeDifference = theano.tensor.dot(probabilidad_Vk.T, probabilidad_Hk)

        delta = positiveDifference - negativeDifference
        wUpdate = momentum * self.vishidinc
        wUpdate += lr_pesos * delta / miniBatchSize
        #wUpdate *= (1.0/(dropout))
        updates.append((self.w, self.w + wUpdate))
        updates.append((self.vishidinc, wUpdate))

        # actualizacion de los bias visibles
        visibleBiasDiff = theano.tensor.sum(self.x - muestra_Vk , axis=0)
        biasVisUpdate = momentum * self.visbiasinc
        biasVisUpdate += lr_vbias * visibleBiasDiff / miniBatchSize
        updates.append((self.visbiases, self.visbiases + biasVisUpdate))
        updates.append((self.visbiasinc, biasVisUpdate))

        # actualizacion de los bias ocultos
        hiddenBiasDiff = theano.tensor.sum(probabilidad_H0 - probabilidad_Hk, axis=0)
        biasHidUpdate = momentum * self.hidbiasinc
        biasHidUpdate += lr_hbias * hiddenBiasDiff / miniBatchSize
        updates.append((self.hidbiases, self.hidbiases + biasHidUpdate))
        updates.append((self.hidbiasinc, biasHidUpdate))

        # se concatenan todas las actualizaciones en una sola
        updates += updatesOriginal.items()

        return updates

    def buildUpdates_bengio(self, updates, cost, constant):
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


    def entrenamiento(self, data, tamMiniBatch=10, tamMacroBatch=None, pcd=True, gibbsSteps=1, validationData=None, tamMacroBatchVal=None, filtros=False, printCompacto=False):
        """
        proceso central de entrenamiento.

        :type tamMiniBatch: int
        :param tamMiniBatch: cantidad de ejeemplos del subconjunto
        """
        memoria_dataset, memoria_por_ejemplo, memoria_por_minibatch = calcular_memoria_requerida(cantidad_ejemplos=data.shape[0], cantidad_valores=data.shape[1], tamMiniBatch=tamMiniBatch)


        if tamMacroBatch is None:
            tamMacroBatch = calcular_chunk(memoriaDatos=memoria_dataset, tamMiniBatch=tamMiniBatch, cantidadEjemplos=data.shape[0])

        tamMacroBatch = int(tamMacroBatch)
        tamMiniBatch = int(tamMiniBatch)

        # comprobaciones por si las dudas
        assert data.shape[0] % tamMacroBatch == 0
        assert tamMacroBatch % tamMiniBatch == 0

        macro_batch_count = int(data.shape[0] / tamMacroBatch)
        micro_batch_count = int(tamMacroBatch / tamMiniBatch)

        # buffer para alojar los datos de entrenamiento
        sharedData = theano.shared(numpy.empty((tamMacroBatch,) + data.shape[1:], dtype=theanoFloat), borrow=True)

        # para la validacion
        if validationData is not None:
            tamMacroBatchVal = validationData.shape[0] if tamMacroBatchVal is None else tamMacroBatchVal
            macro_batch_val_count = int(validationData.shape[0] / tamMacroBatchVal)
            sharedDataValidation = theano.shared(numpy.empty((tamMacroBatchVal,) + validationData.shape[1:], dtype=theanoFloat), borrow=True)


        """
        self.sharedEstadisticos = theano.shared(numpy.zeros((10), dtype=theanoFloat))
        """


        trainer = None
        self.fnActivacionUnidEntrada = self.params['unidadesVisibles']
        self.fnActivacionUnidSalida = self.params['unidadesOcultas']
        if pcd:
            print("Entrenando con Divergencia Contrastiva Persistente, {} pasos de Gibss.".format(gibbsSteps))
            trainer = self.DivergenciaContrastivaPersistente(tamMiniBatch, sharedData)
        else:
            print("Entrenando con Divergencia Contrastiva, {} pasos de Gibss.".format(gibbsSteps))
            trainer = self.DivergenciaContrastiva(tamMiniBatch, sharedData)
        print("Unidades de visibles:",self.fnActivacionUnidEntrada, "Unidades Ocultas:", self.fnActivacionUnidSalida)

        if filtros:
            # plot los filtros iniciales (sin entrenamiento)
            self.dibujarFiltros(nombreArchivo='filtros_epoca_0.pdf', automatico=True)


        costo = numpy.Inf; mse = numpy.Inf; fEnergy = numpy.Inf
        finLinea = '\r' if printCompacto else '\n'

        print("Entrenando una RBM, con [{}] unidades visibles y [{}] unidades ocultas".format(self.n_visible, self.n_hidden))
        print("Cantidad de ejemplos para el entrenamiento no supervisado: ", len(data))
        print("Tamanio del MiniBatch: ", tamMiniBatch, "Tamanio MacroBatch: ", tamMacroBatch)
        print("Mem. Disp.:", gpu_info('Mb')[0], "Mem. Dataset:", memoria_dataset, "Mem. x ej.:", memoria_por_ejemplo, "Mem. x Minibatch:", memoria_por_minibatch)

        for epoch in range(1, self.params['epocas']):

            costo = []; mse = []; fEnergy = []
            ####
            # el macro batch se agrega aca, la unica diferencia esta en que se cargan los datos dentro
            # y se se ejecuta normalmente
            ####
            for macro_batch_index in range(0,macro_batch_count):
            ###
                sharedData.set_value(data[macro_batch_index * tamMacroBatch: (macro_batch_index + 1) * tamMacroBatch], borrow=True)

                for batch in range(0, micro_batch_count):
                    # salida[monitoring_cost, mse, deltafreeEnergy]
                    salida = trainer(batch, gibbsSteps)

                    costo.append(salida[0])
                    mse.append(salida[1])
                    fEnergy.append(salida[2])

            ###

            costo = numpy.mean(costo)
            mse = numpy.mean(mse)
            fEnergy = numpy.mean(fEnergy)
            """
            costo = self.sharedEstadisticos.get_value()[0] / micro_batch_count
            mse = self.sharedEstadisticos.get_value()[1] / micro_batch_count
            fEnergy = self.sharedEstadisticos.get_value()[2] / micro_batch_count
            self.sharedEstadisticos.set_value([0.0]*10)
            """

            self.agregarEstadistico(
                {'errorEntrenamiento': costo,
                 'mseEntrenamiento': mse,
                 'errorValidacion': 0.0,
                 'errorTesteo': 0.0,
                 'energiaLibreEntrenamiento': fEnergy,
                 'energiaLibreValidacion': 0.0})

            # imprimo algo de informacion sobre la terminal
            print(str('Epoca {:>3d} de {:>3d}, error<TrnSet>:{:> 8.5f}, MSE<ejemplo>:{:> 8.5f}, EnergiaLibre<ejemplo>:{:> 8.5f}').format(
                        epoch, self.params['epocas'], costo, mse, fEnergy),
                    end=finLinea)

            if filtros:
                self.dibujarFiltros(nombreArchivo='filtros_epoca_{}.pdf'.format(epoch), automatico=True)

            # END SET
        # END epoch
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior

        self.dibujarEstadisticos()
        return 1

    def reconstruccion(self, muestraV, gibbsSteps=1):
        """
        realiza la reconstruccion a partir de un ejemplo, efectuando una cadena
        de markov con x pasos de gibbs sobre la misma

        puedo pasarle un solo ejemplo o una matrix con varios de ellos por fila
        """
        warn("esta implementacion no retorna la probabilidad de activacion de las unidades ocultas, por eso queda deprecated, utilizar muestra")

        if muestraV.ndim == 1:
            # es un vector, debo cambiar el root 'x' antes de armar el grafo
            # para que coincida con la entrada
            viejoRoot = self.x
            self.x = theano.tensor.fvector('x')

        data  = theano.shared(numpy.asarray(a=muestraV,dtype=theanoFloat), name='datoReconstruccion')

        ([_, _, muestra_H1, _, probabilidad_V1, muestra_V1],
          updates) = self.gibbsVHV(data, gibbsSteps)

        reconstructor = theano.function(
                        inputs=[],
                        outputs=[probabilidad_V1[-1], muestra_V1[-1], muestra_H1[-1]],
                        updates=updates,
                        #givens={self.x: data},
                        name='reconstructor'
        )

        [probabilidad_V1, muestra_V1, muestra_H1] = reconstructor()

        if muestraV.ndim == 1:
            # hago el swap
            self.x = viejoRoot

        return [probabilidad_V1, muestra_V1, muestra_H1]

    def muestra(self, muestraV, gibbsSteps=1):
        """
        realiza la reconstruccion a partir de un ejemplo, efectuando una cadena
        de markov con n pasos de gibbs sobre la misma

        puedo pasarle un solo ejemplo o una matrix con varios de ellos por fila
        """

        if muestraV.ndim == 1:
            # es un vector, debo cambiar el root 'x' antes de armar el grafo
            # para que coincida con la entrada
            viejoRoot = self.x
            self.x = theano.tensor.fvector('x')

        data  = theano.shared(numpy.asarray(a=muestraV, dtype=theanoFloat), borrow=True, name='datoMuestra')

        ([salidaLineal_H1, probabilidad_H1, muestra_H1, salidaLineal_V1, probabilidad_V1, muestra_V1], updates) = self.gibbsVHV(data, gibbsSteps)

        reconstructor = theano.function(
                        inputs=[],
                        outputs=[probabilidad_H1[-1], muestra_V1[-1], muestra_H1[-1]],
                        updates=updates,
                        #givens={self.x: data},
                        name='fn_muestra'
        )

        [probabilidad_H1, muestra_V1, muestra_H1] = reconstructor()

        if muestraV.ndim == 1:
            # hago el swap
            self.x = viejoRoot

        del data
        return [probabilidad_H1, muestra_V1, muestra_H1]


    def sampleo(self, data, labels=None, chains=20, samples=10, gibbsSteps=1000, patchesDim=(28,28), binary=False):
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

        # inicializacion de todas las cadenas... el estado persiste a traves
        # del muestreo
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

        ([_, _, _, _, probabilidad_V1, muestra_V1],
          updates) = self.gibbsVHV(self.x, gibbsSteps)

        # cambiar el valor de la cadenaFija al de la reconstruccion de las visibles
        updates.update({cadenaFija: muestra_V1[-1]})

        # funcion princial
        muestreo = theano.function(inputs=[],
                                   outputs=[probabilidad_V1[-1], muestra_V1[-1]],
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
            probabilidad_V1, visiblerecons = muestreo()

            print(' ... plotting sample {}'.format(idx))
            imageResults[(alto+1) * idx:(ancho+1) * idx + ancho, :] \
                = imagenTiles(X=probabilidad_V1,
                              img_shape=(alto, ancho),
                              tile_shape=(1, chains),
                              tile_spacing=(1, 1)
                )

        # construct image
        image = Image.fromarray(imageResults)

        nombreArchivo = self.ruta + "samples_" + str(labels[lista])
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
        warn("esta funcion sera <<deprecated>>, ver por guardarPrametros etc")
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

    def guardarParametros(self, nombreArchivo):
        """
        guarda los parametros de la red en formato comprimido
        """
        numpy.savez_compressed(self.ruta + nombreArchivo + '.npz', w=self.get_w, visBias=self.get_biasVisible, hidBias=self.get_biasOculto)
        return 1

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
