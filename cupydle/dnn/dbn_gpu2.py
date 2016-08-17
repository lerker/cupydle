#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implelemtacion de una red de creencia profunda en GPU"""

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


import numpy
import os

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

from cupydle.dnn.funciones import sigmoideaTheano


"""




"""
try:
    import PIL.Image as Image
except ImportError:
    import Image

from cupydle.dnn.graficos import scale_to_unit_interval
from cupydle.dnn.graficos import imagenTiles

import matplotlib.pyplot

# parte de las rbm
from cupydle.dnn.rbm_gpu2 import RBM

from cupydle.dnn.utils import save
from cupydle.dnn.utils import load as load_utils
from cupydle.dnn.utils import temporizador
from cupydle.dnn.mlp import MLP

import glob# load_dbn_weight

class rbmParams(object):
    # sirve para guardar el estado nomas
    def __init__(self,
                n_visible,
                n_hidden,
                numEpoch,
                batchSize,
                epsilonw,
                pasosGibbs=1,
                w=None,
                epsilonvb=None,
                epsilonhb=None,
                weightcost=None,
                momentum=0.0):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        self.numEpoch=numEpoch
        self.epsilonw=epsilonw
        self.pasosGibbs=pasosGibbs
        self.w=w
        self.epsilonvb=epsilonvb
        self.epsilonhb=epsilonhb
        self.weightcost=weightcost
        self.momentum=momentum
        self.batchSize=batchSize

        return

    def __str__(self):
        print("Numero de neuronas visibles:",self.n_visible)
        print("Numero de neuronas ocultas:",self.n_hidden)
        print("Numero de epocas:",self.numEpoch)
        print("Tasa de aprendizaje para los pesos:",self.epsilonw)
        print("Tasa de aprendizaje para las unidades visibles",self.epsilonvb)
        print("Tasa de aprendizaje para las unidades ocultas:",self.epsilonhb)
        print("Tasa de momento:",self.momentum)
        print("Castigo pesos:",self.weightcost)
        print("Pasos de Gibss:", self.pasosGibbs)
        print("Tamaño del batch:",self.batchSize)
        return str("")

    #def __call__(self):
    #    return

class DBN(object):

    verbose = True

    def __init__(self, numpy_rng=None, theano_rng=None, n_outs=None, name=None, ruta=''):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network (for clasification)

        :type name: string
        :param name: name of model, for save files in disk
        """

        if not numpy_rng:
            numpy_rng = numpy.random.RandomState(1234)
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # si es de clasificacion, n_outs es distinto a none, es un numero de clases
        self.n_outs = n_outs

        # allocate symbolic variables for the data
        self.x = theano.tensor.matrix('samples')  # the data is presented as rasterized images
        self.y = theano.tensor.ivector('labels')  # the labels are presented as 1D vector
                                                  # of [int] labels

        self.params = []
        self.n_layers = 0
        self.layersHidAct = [] # activaciones de las capas ocultas intermedias
        self.pesos = [] #pesos almacenados en una lista, una vez entrenados se guardan aca
                        # para utilizarse en el fit o prediccion...

        if name is None:
            name="dbnTest"
        self.name = name

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        self.ruta = ruta

        # parametros para el ajuste fino o por medio de un mlp
        self.parametrosAjuste = {}
        self._initParametrosAjuste()

    def _initParametrosAjuste(self):
        """ inicializa los parametros de la red, un diccionario"""
        self.parametrosAjuste['tasaAprendizaje'] = 0.0
        self.parametrosAjuste['regularizadorL1'] = 0.0
        self.parametrosAjuste['regularizadorL2'] = 0.0
        self.parametrosAjuste['momento'] = 0.0
        self.parametrosAjuste['epocas'] = 0.0
        self.parametrosAjuste['activationfuntion'] = sigmoideaTheano()
        return 1

    def setParametrosAjuste(self, parametros):
        if not isinstance(parametros, dict):
            assert False, "necesito un diccionario"

        for key, _ in parametros.items():
            if key in self.parametrosAjuste:
                self.parametrosAjuste[key] = parametros[key]
            else:
                assert False, "la clave(" + str(key) + ") en la variable paramtros no existe"
        return 1


    def addLayer(self,
                n_hidden,
                numEpoch,
                batchSize,
                epsilonw,        # Learning rate for weights
                n_visible=None,
                pasosGibbs=1,
                w=None,
                epsilonvb=None,  # Learning rate for biases of visible units
                epsilonhb=None,  # Learning rate for biases of hidden units
                weightcost=None, # Weigth punishment
                momentum=0.0):

        if n_visible is None:
            assert self.params != [], "Debe prover de unidades de entrada para esta capa."
            n_visible = self.params[-1].n_hidden

        # agrego una capa de rbm a la dbn, con los parametros que le paso
        self.params.append(rbmParams(n_visible=n_visible, n_hidden=n_hidden,
                                    numEpoch=numEpoch, batchSize=batchSize,
                                    epsilonw=epsilonw, pasosGibbs=pasosGibbs,
                                    w=w, epsilonvb=epsilonvb, epsilonhb=epsilonhb,
                                    weightcost=weightcost, momentum=momentum))
        self.n_layers += 1

        return

    def preEntrenamiento(self, dataTrn, dataVal, pcd=True,
                         guardarPesosIniciales=False, filtros=True):
        """
        :type dataTrn: narray
        :param dataTrn: datos de entrenamiento

        :type dataVal: narray
        :param dataTrn: datos de validacion

        :type guardarPesosIniciales: boolean
        :param guardarPesosIniciales: si se requiere almacenar los pesos iniciales antes de aplicar la rbm
        """
        assert self.n_layers > 0, "No hay nada que computar"
        self.x = dataTrn

        for i in range(self.n_layers):
            # deben diferenciarse si estamos en presencia de la primer capa o de una intermedia
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layersHidAct[-1]

            # una carpeta para alojar los datos
            directorioRbm = self.ruta + 'rbm_capa{}/'.format(i+1)
            if not os.path.exists(directorioRbm):
                os.makedirs(directorioRbm)


            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(n_visible=self.params[i].n_visible,
                            n_hidden=self.params[i].n_hidden,
                            w=self.params[i].w,
                            ruta=directorioRbm)

            # configuro la capa, la rbm
            rbm_layer.setParams({'epsilonw':self.params[i].epsilonw})
            rbm_layer.setParams({'epsilonvb':self.params[i].epsilonvb})
            rbm_layer.setParams({'epsilonhb':self.params[i].epsilonhb})
            rbm_layer.setParams({'momentum':self.params[i].momentum})
            rbm_layer.setParams({'weightcost':self.params[i].weightcost})
            rbm_layer.setParams({'maxepoch':self.params[i].numEpoch})
            # activation function?

            # train it!! layer per layer
            print("Entrenando la capa:", i+1)
            if guardarPesosIniciales:
                nombre = self.name + "_pesosInicialesCapa" + str(i+1)
                rbm_layer.guardarPesos(nombreArchivo=nombre)

            rbm_layer.entrenamiento(data=layer_input,   # los datos los binarizo y convierto a float
                            miniBatchSize=self.params[i].batchSize,
                            pcd=pcd,
                            gibbsSteps=self.params[i].pasosGibbs,
                            validationData=dataVal,
                            filtros=filtros)

            print("Guardando la capa..") if DBN.verbose else None
            filename = self.name + "_capa" + str(i+1) + ".pgz"
            rbm_layer.guardar(nombreArchivo=filename)

            # ahora debo tener las entras que son las salidas del modelo anterior (activaciones de las ocultas)
            # TODO aca debo tener las activaciones de las ocultas
            [_, _, hiddenActPos] = rbm_layer.reconstruccion(muestraV=layer_input)
            [_, _, dataVal] = rbm_layer.reconstruccion(muestraV=dataVal)
            #hiddenActPos = rbm_layer.activacionesOcultas(layer_input)
            #dataVal = rbm_layer.activacionesOcultas(dataVal)

            print("Guardando las muestras para la siguiente capa..") if DBN.verbose else None
            filename_samples = self.ruta + self.name + "_ActivacionesOcultas" + str(i+1)
            save(objeto=hiddenActPos, filename=filename_samples, compression='gzip')

            # guardo la salida de la capa para la proxima iteracion
            self.layersHidAct.append(hiddenActPos)

            del rbm_layer
        # FIN FOR

        # una vez terminado el entrenamiento guardo los pesos para su futura utilizacion
        self.pesos = self.cargarPesos(dbnNombre=self.name, ruta=self.ruta)
        return
    # FIN TRAIN

    def ajuste(self, datos, listaPesos=None, fnActivacion=sigmoideaTheano(),
               semillaRandom=None):
        """
        construye un perceptron multicapa, y ajusta los pesos por medio de un
        entrenamiento supervisado.
        """
        if listaPesos is None:
            if self.pesos == []:
                self.pesos = self.cargarPesos(dbnNombre=self.name, ruta=self.ruta)
        else:
            self.pesos = listaPesos

        assert self.pesos!=[], "Error al obtener los pesos"

        activaciones = []
        if fnActivacion is not None:
            if isinstance(fnActivacion, list):
                assert len(fnActivacion) == len(self.pesos), "No son suficientes funciones de activacion"
                activaciones = fnActivacion
            else:
                activaciones = [fnActivacion] * len(self.pesos)
        else:
            assert False, "Debe proveer una funcion de activacion"


        clasificador = MLP(clasificacion=True,
                           rng=semillaRandom,
                           ruta=self.ruta)
        """
        clasificador.setParametroEntrenamiento({'tasaAprendizaje':self.parametrosAjuste['tasaAprendizaje']})
        clasificador.setParametroEntrenamiento({'regularizadorL1':self.parametrosAjuste['regularizadorL1']})
        clasificador.setParametroEntrenamiento({'regularizadorL2':self.parametrosAjuste['regularizadorL2']})
        clasificador.setParametroEntrenamiento({'momento':self.parametrosAjuste['momento']})
        clasificador.setParametroEntrenamiento({'epocas':self.parametrosAjuste['epocas']})
        clasificador.setParametroEntrenamiento({'activationfuntion':self.parametrosAjuste['activationfuntion']})
        """
        clasificador.setParametroEntrenamiento(self.parametrosAjuste)

        # cargo en el perceptron multicapa los pesos en cada capa
        # como el fit es de clasificacion, las primeras n-1 capas son del tipo
        # 'logisticas' luego la ultima es un 'softmax'
        for i in range(0,len(self.pesos)-1):
            clasificador.agregarCapa(unidadesEntrada=self.pesos[i].shape[0],
                                     unidadesSalida=self.pesos[i].shape[1],
                                     clasificacion=False,
                                     activacion=activaciones[i],
                                     pesos=self.pesos[i],
                                     biases=None)

        clasificador.agregarCapa(unidadesEntrada=self.pesos[-1].shape[0],
                                 unidadesSalida=self.pesos[-1].shape[1],
                                 clasificacion=True,
                                 activacion=activaciones[-1],
                                 pesos=self.pesos[-1],
                                 biases=None)

        T = temporizador()
        inicio = T.tic()

        clasificador.train(
                        trainSet=datos[0],
                        validSet=datos[1],
                        testSet=datos[2],
                        batch_size=10)

        final = T.toc()
        print("Tiempo total para entrenamiento DBN: {}".format(T.transcurrido(inicio, final)))
        return



    def save2(self, filename, compression='zip'):
        """
        guarda la dbn, algunos datos para poder levantarla
        """
        datos = {"name":self.name,
                    "layers":self.layers,
                    "n_layers":self.n_layers,
                    "params":self.params,
                    "numpy_rng":self.numpy_rng,
                    "theano_rng":self.theano_rng}
        save(objeto=datos, filename=filename, compression=compression)
        return

    def save(self, filename, compression='zip'):
        """
        guarda la dbn, algunos datos para poder levantarla
        """

        save(objeto=self, filename=filename, compression=compression)
        return

    @staticmethod
    def load(filename, compression='zip'):
        datos = None
        datos = load_utils(filename=filename, compression=compression)
        return datos

    # redefino
    def __str__(self):
        print("Name:", self.name)
        print("Cantidad de capas:", self.n_layers)
        for i in range(0,len(self.params)):
            print("-[" + str(i+1) + "] :")
            print(self.params[i])
        #if self.clasificacion:
        #    print("DBN para la tarea de clasificacion")
        #else:
        #    print("DBN para la tarea reconstruccion")
        return str("")

    @property
    def info(self):
        print("Name:", self.name)
        print("Cantidad de capas:", self.n_layers)
        return 0

    @staticmethod
    def cargarPesos(dbnNombre, ruta):
        """
        carga las primeras 10 capas de la dbn segun su nombre y directorio
        (ordenadas por nombre)
        """
        capas = []
        for directorio in sorted(glob.glob(ruta+'rbm_capa[0-9]/')):
            for file in sorted(glob.glob(directorio + dbnNombre + "_capa[0-9].*")):
                print("Cargando capa: ",file) if DBN.verbose else None
                capas.append(RBM.load(str(file)).w.get_value())
        return capas

if __name__ == "__main__":
    assert False, str(__file__ + " No es un modulo")
