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

# sistema basico
# sistema basico
import time # tiempo requedo por iteracion, se puede eliminar
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

# parte de las rbm
from cupydle.dnn.rbm_gpu2 import RBM

from cupydle.dnn.utils import save
from cupydle.dnn.utils import load as load_utils

import glob# load_dbn_weight

class rbmParams(object):
    # sirve para guardar el estado nomas
    def __init__(self,
                n_visible,
                n_hidden,
                numEpoch,
                batchSize,
                epsilonw,        # Learning rate for weights
                w=None,
                epsilonvb=None,  # Learning rate for biases of visible units
                epsilonhb=None,  # Learning rate for biases of hidden units
                weightcost=None, # Weigth punishment
                initialmomentum=None,
                finalmomentum=None):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        self.numEpoch=numEpoch
        self.epsilonw=epsilonw
        self.w=w
        self.epsilonvb=epsilonvb
        self.epsilonhb=epsilonhb
        self.weightcost=weightcost
        self.initialmomentum=initialmomentum
        self.finalmomentum=finalmomentum
        self.batchSize=batchSize

        return

    def __str__(self):
        print("Numero de neuronas visibles:",self.n_visible)
        print("Numero de neuronas ocultas:",self.n_hidden)
        print("Numero de epocas:",self.numEpoch)
        print("Tasa de aprendizaje para los pesos:",self.epsilonw)
        print("Tasa de aprendizaje para las unidades visibles",self.epsilonvb)
        print("Tasa de aprendizaje para las unidades ocultas:",self.epsilonhb)
        print("Castigo pesos:",self.weightcost)
        print("Tasa de momento inicial:",self.initialmomentum)
        print("Tasa de momento final:",self.finalmomentum)
        print("Tamanio del batch:",self.batchSize)
        return str("")

    #def __call__(self):
    #    return

class dbn(object):

    def __init__(self, numpy_rng=None, theano_rng=None, n_outs=None, name=None):
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
        self.params = []

        self.n_layers = 0

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
        self.layers = []

        if name is None:
            name="dbnTest"
        self.name = name

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

    def addLayer(self,
                n_visible,
                n_hidden,
                numEpoch,
                batchSize,
                epsilonw,        # Learning rate for weights
                w=None,
                epsilonvb=None,  # Learning rate for biases of visible units
                epsilonhb=None,  # Learning rate for biases of hidden units
                weightcost=None, # Weigth punishment
                initialmomentum=None,
                finalmomentum=None):
        # agrego una capa de rbm a la dbn, con los parametros que le paso
        self.params.append(rbmParams(n_visible=n_visible, n_hidden=n_hidden,
                                    numEpoch=numEpoch, batchSize=batchSize,
                                    epsilonw=epsilonw, w=w, epsilonvb=epsilonvb,
                                    epsilonhb=epsilonhb, weightcost=weightcost,
                                    initialmomentum=initialmomentum, finalmomentum=finalmomentum))
        self.n_layers += 1

        return

    def train(self, dataTrn, dataVal, path='./', saveInitialWeights=False):
        """
        :type dataTrn: narray
        :param dataTrn: datos de entrenamiento

        :type dataVal: narray
        :param dataTrn: datos de validacion

        :type path: string
        :param path: directorio donde almacenar los resultados, archivos

        :type saveInitialWeigths: boolean
        :param saveInitialWeights: si se requiere almacenar los pesos iniciales antes de aplicar la rbm
        """
        assert self.n_layers > 0, "No hay nada que computar"
        self.x = dataTrn
        filtrosss = path

        for i in range(self.n_layers):
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layers[-1]
                filtrosss=None

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(n_visible=self.params[i].n_visible,
                            n_hidden=self.params[i].n_hidden,
                            w=self.params[i].w)

            # configuro la capa, la rbm
            rbm_layer.setParams({'epsilonw':self.params[i].epsilonw})
            rbm_layer.setParams({'epsilonvb':self.params[i].epsilonvb})
            rbm_layer.setParams({'epsilonhb':self.params[i].epsilonhb})
            rbm_layer.setParams({'initialmomentum':self.params[i].initialmomentum})
            rbm_layer.setParams({'weightcost':self.params[i].weightcost})
            rbm_layer.setParams({'maxepoch':self.params[i].numEpoch})
            # activation function?

            # train it!! layer per layer
            print("Entrenando la capa:", i+1)
            if saveInitialWeights:
                filename = path + self.name + "_capaInicial" + str(i+1) + ".pgz"
                rbm_layer.save(filename)

            rbm_layer.train(    data=layer_input,   # los datos los binarizo y convierto a float
                                miniBatchSize=self.params[i].batchSize,
                                pcd=True,
                                gibbsSteps=15,
                                validationData=dataVal,
                                plotFilters=filtrosss)

            print("Guardando la capa..") if verbose else None
            filename = path + self.name + "_capa" + str(i+1) + ".pgz"
            rbm_layer.save(filename, absolutName=True)

            # ahora debo tener las entras que son las salidas del modelo anterior (activaciones de las ocultas)
            # TODO aca debo tener las activaciones de las ocultas
            hiddenActPos = rbm_layer.activacionesOcultas(layer_input)
            dataVal = rbm_layer.activacionesOcultas(dataVal)

            print("Guardando las muestras para la siguiente capa..") if verbose else None
            filename_samples = path + self.name + "_capaSample" + str(i+1)
            save(objeto=hiddenActPos, filename=filename_samples, compression='gzip')

            # guardo la salida de la capa para la proxima iteracion
            self.layers.append(hiddenActPos)

            del rbm_layer
        # FIN FOR
        return
    # FIN TRAIN

    def fit(self, lista_pesos, mn, n_epochs=1000):
        classifier = MLP(   task="clasificacion",
                            rng=None)

        classifier.addLayer(
                            unitsIn=784,
                            unitsOut=1000,
                            classification=False,
                            activation=Sigmoid(),
                            weight=lista_pesos[0],
                            bias=None)

        classifier.addLayer(
                            unitsIn=1000,
                            unitsOut=1000,
                            classification=False,
                            weight=lista_pesos[1],
                            bias=None)

        classifier.addLayer(
                            unitsIn=1000,
                            unitsOut=10,
                            classification=True,
                            activation=Sigmoid(),
                            weight=lista_pesos[2],
                            bias=None)

        T = timer2()
        inicio = T.tic()

        classifier.train(
                        trainSet=mn.get_training(),
                        validSet=mn.get_validation(),
                        testSet=mn.get_testing(),
                        batch_size=10,
                        n_epochs=n_epochs)

        final = T.toc()
        print("Tiempo total para entrenamiento DBN: {}".format(T.elapsed(inicio, final)))
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

def load_dbn_weigths(path, dbnName):
    """
    carga las primeras 10 capas de la dbn segun su nombre y directorio (ordenadas por nombre)
    http://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered
    """
    capas = []
    for file in sorted(glob.glob(path + dbnName + "_capa[0-9].*")):
        print("Cargando capa: ",file) if verbose else None
        capas.append(RBM.load(str(file)))
    return capas

if __name__ == "__main__":
    from cupydle.dnn.utils import timer as timer2
    import os
    currentPath = os.getcwd()                               # directorio actual de ejecucion
    testPath    = currentPath + '/cupydle/test/mnist/'      # sobre el de ejecucion la ruta a los tests
    dataPath    = currentPath + '/cupydle/data/DB_mnist/'   # donde se almacenan la base de datos
    testFolder  = 'test1/'                                  # carpeta a crear para los tests
    fullPath    = testPath + testFolder

    if not os.path.exists(fullPath):        # si no existe la crea
        print('Creando la carpeta para el test en: ',fullPath)
        os.makedirs(fullPath)

    import subprocess
    subprocess.call(testPath + 'get_data.sh', shell=True)   # chequeo si necesito descargar los datos

    from cupydle.test.mnist.mnist import MNIST
    setName = "mnist"
    MNIST.prepare(dataPath, nombre=setName, compresion='bzip2')

    import argparse
    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre MNIST.')
    parser.add_argument('-r', '--rbm', action="store_true", dest="rbm", help="ejecutar las rbm", default=False)
    parser.add_argument('-m', '--mlp', action="store_true", dest="mlp", help="mlp simple", default=False)
    parser.add_argument('-d', '--dbn', action="store_true", dest="dbn", help="dbn", default=False)
    parser.add_argument('-e', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    rbm = args.rbm
    mlp = args.mlp
    dbnn = args.dbn
    modelName = args.modelo
    #modelName = 'capa1.pgz'
    verbose = True

    if mlp :
        print("S E C C I O N        M L P")

        # Dependencias Externas
        from cupydle.dnn.mlp import MLP
        from cupydle.test.mnist.mnist import MNIST
        from cupydle.test.mnist.mnist import open4disk
        from cupydle.test.mnist.mnist import save2disk


        # se leen de disco la base de datos
        mn = open4disk(filename=dataPath + setName, compression='bzip2')
        #mn.info                                        # muestra alguna informacion de la base de datos

        classifier = MLP(   task="clasificacion",
                            rng=None)

        classifier.addLayer(
                            unitsIn=784,
                            unitsOut=1000,
                            classification=False,
                            activation=Sigmoid(),
                            weight=None,
                            bias=None)

        classifier.addLayer(
                            unitsIn=1000,
                            unitsOut=1000,
                            classification=False,
                            weight=None,
                            bias=None)

        classifier.addLayer(
                            unitsIn=1000,
                            unitsOut=10,
                            classification=True,
                            activation=Sigmoid(),
                            weight=None,
                            bias=None)

        T = timer2()
        inicio = T.tic()

        numpy.save(fullPath + "pesos1",classifier.capas[0].W.get_value())
        numpy.save(fullPath + "pesos2",classifier.capas[1].W.get_value())
        numpy.save(fullPath + "pesos3",classifier.capas[2].W.get_value())

        classifier.train(
                        trainSet=mn.get_training(),
                        validSet=mn.get_validation(),
                        testSet=mn.get_testing(),
                        batch_size=10,
                        n_epochs=1000)

        final = T.toc()
        print("Tiempo total para entrenamiento MLP: {}".format(T.elapsed(inicio, final)))

    if rbm :
        print("S E C C I O N        R B M")

        from cupydle.test.mnist.mnist import MNIST
        from cupydle.test.mnist.mnist import open4disk
        from cupydle.test.mnist.mnist import save2disk

        # se leen de disco los datos
        mnData = open4disk(filename=dataPath + setName, compression='bzip2')
        #mn.info  # muestra alguna informacion de la base de datos

        # obtengo todos los subconjuntos
        train_img,  train_labels= mnData.get_training()
        test_img,   test_labels = mnData.get_testing()
        val_img,    val_labels  = mnData.get_validation()

        dbn0 = dbn(name=None)

        pesos1 = numpy.load(fullPath + "pesos1.npy")
        pesos2 = numpy.load(fullPath + "pesos2.npy")
        pesos3 = numpy.load(fullPath + "pesos3.npy")

        # agrego una capa..
        dbn0.addLayer(n_visible=784,
                      n_hidden=1000,
                      numEpoch=100,
                      batchSize=10,
                      epsilonw=0.01,
                      w=pesos1)
        # otra capa mas
        dbn0.addLayer(n_visible=1000, # coincide con las ocultas de las anteriores
                      n_hidden=1000,
                      numEpoch=100,
                      batchSize=10,
                      epsilonw=0.01,
                      w=pesos2)

        # clasificacion
        dbn0.addLayer(n_visible=1000, # coincide con las ocultas de las anteriores
                      n_hidden=10,
                      numEpoch=100,
                      batchSize=10,
                      epsilonw=0.01,
                      w=pesos3)

        T = timer2()
        inicio = T.tic()

        #entrena la red
        dbn0.train(dataTrn=(train_img/255.0).astype(numpy.float32),
                   dataVal=(val_img/255.0).astype(numpy.float32),
                   path=fullPath)

        final = T.toc()
        print("Tiempo total para entrenamiento DBN: {}".format(T.elapsed(inicio, final)))

        dbn0.save(fullPath + "dbnMNIST", compression='zip')

    if dbnn:
        print("S E C C I O N        D B N")
        miDBN = dbn.load(filename=fullPath + "dbnMNIST", compression='zip')
        print(miDBN)

        red_capas = load_dbn_weigths(fullPath, miDBN.name)

        # Dependencias Externas
        from cupydle.dnn.mlp import MLP
        from cupydle.test.mnist.mnist import MNIST
        from cupydle.test.mnist.mnist import open4disk
        from cupydle.test.mnist.mnist import save2disk


        # se leen de disco la base de datos
        mn = open4disk(filename=dataPath + setName, compression='bzip2')
        #mn.info                                        # muestra alguna informacion de la base de datos

        # obtengo todos los subconjuntos
        train_img,  train_labels= mn.get_training()
        test_img,   test_labels = mn.get_testing()
        val_img,    val_labels  = mn.get_validation()

        #lista_pesos = [rbm.w.get_value().transpose() for rbm in red_capas]
        lista_pesos = [rbm.w.get_value() for rbm in red_capas]

        miDBN.fit(lista_pesos, mn, n_epochs=1000)



