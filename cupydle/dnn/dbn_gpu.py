#!/usr/bin/env python3

# TODO
# explota por falta de memoria, se ve que algo que hice del free energy
# Error en el NeuralNetwork tipo de error MSE, con CROSSENTROPY funciona

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

# parte de las rbm
from cupydle.dnn.rbm_gpu import rbm_gpu

from cupydle.dnn.utils import save
from cupydle.dnn.utils import load as load_utils

import glob# load_dbn_weight

verbose = False

class rbmParams(object):
    # sirve para guardar el estado nomas
    def __init__(self,
                n_visible,
                n_hidden,
                numEpoch,
                batchSize,
                epsilonw,        # Learning rate for weights
                epsilonvb=None,  # Learning rate for biases of visible units
                epsilonhb=None,  # Learning rate for biases of hidden units
                weightcost=None, # Weigth punishment
                initialmomentum=None,
                finalmomentum=None):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        self.numEpoch=numEpoch
        self.epsilonw=epsilonw
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
                epsilonvb=None,  # Learning rate for biases of visible units
                epsilonhb=None,  # Learning rate for biases of hidden units
                weightcost=None, # Weigth punishment
                initialmomentum=None,
                finalmomentum=None):
        # agrego una capa de rbm a la dbn, con los parametros que le paso
        self.params.append(rbmParams(n_visible=n_visible, n_hidden=n_hidden,
                                    numEpoch=numEpoch, batchSize=batchSize,
                                    epsilonw=epsilonw, epsilonvb=epsilonvb,
                                    epsilonhb=epsilonhb, weightcost=weightcost,
                                    initialmomentum=initialmomentum, finalmomentum=finalmomentum))
        self.n_layers += 1

        return

    def train(self, dataTrn, dataVal, path='./', saveInitialLayer=False):
        """
        :type dataTrn: narray
        :param dataTrn: datos de entrenamiento

        :type dataVal: narray
        :param dataTrn: datos de validacion

        :type path: string
        :param path: directorio donde almacenar los resultados, archivos
        """
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.
        assert self.n_layers > 0, "No hay nada que computar"
        self.x = dataTrn

        for i in range(self.n_layers):
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layers[-1]

            # Construct an RBM that shared weights with this layer
            rbm_layer = rbm_gpu(n_visible=self.params[i].n_visible,
                                n_hidden=self.params[i].n_hidden)

            # configuro la capa, la rbm
            rbm_layer.setParams(numEpoch =          self.params[i].numEpoch,
                                epsilonw =          self.params[i].epsilonw,
                                epsilonvb =         self.params[i].epsilonvb,
                                epsilonhb =         self.params[i].epsilonhb,
                                weightcost =        self.params[i].weightcost,
                                initialmomentum =   self.params[i].initialmomentum,
                                finalmomentum =     self.params[i].finalmomentum,
                                activationFunction =None)

            # train it!! layer per layer
            print("Entrenando la capa:", i+1)
            if saveInitialLayer:
                filename = path + self.name + "_capaInicial" + str(i+1) + ".pgz"
                rbm_layer.save(filename)

            rbm_layer.train(data =          layer_input,
                            miniBatchSize = self.params[i].batchSize,
                            validationData =dataVal)

            print("Guardando la capa..") if verbose else None
            filename = path + self.name + "_capa" + str(i+1) + ".pgz"
            rbm_layer.save(filename, absolutName=True)

            # ahora debo tener las entras que son las salidas del modelo anterior (activaciones de las ocultas)
            # TODO aca debo tener las activaciones de las ocultas
            [_, hiddenActPos, _, _] = rbm_layer.sampler(layer_input)
            [_, dataVal, _, _] = rbm_layer.sampler(dataVal)

            print("Guardando las muestras para la siguiente capa..") if verbose else None
            filename_samples = path + self.name + "_capaSample" + str(i+1)
            save(objeto=hiddenActPos, filename=filename_samples, compression='gzip')

            # guardo la salida de la capa para la proxima iteracion
            self.layers.append(hiddenActPos)

            del rbm_layer
        # FIN FOR
        return
    # FIN TRAIN

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
        capas.append(rbm_gpu.load(str(file)))
    return capas

if __name__ == "__main__":
    from cupydle.dnn.utils import timer
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
    parser.add_argument('-g', '--guardar', action="store_true", dest="guardar", help="desea guardar (correr desde cero)", default=False)
    parser.add_argument('-m', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    guardar = args.guardar
    modelName = args.modelo
    #modelName = 'capa1.pgz'
    verbose = True

    if guardar :

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

        # binarizacion
        threshold = 128
        binaryTrnData = (train_img>threshold).astype(numpy.float32)
        binaryValData = (val_img>threshold).astype(numpy.float32)

        # TODO solo deberia quedar el nombre
        dbn0 = dbn(name=None)

        # agrego una capa..
        dbn0.addLayer(n_visible=784,
                      n_hidden=500,
                      numEpoch=100,
                      batchSize=20,
                      epsilonw=0.1)
        # otra capa mas
        dbn0.addLayer(n_visible=500, # coincide con las ocultas de las anteriores
                      n_hidden=100,
                      numEpoch=100,
                      batchSize=20,
                      epsilonw=0.1)

        # clasificacion
        dbn0.addLayer(n_visible=100, # coincide con las ocultas de las anteriores
                      n_hidden=10,
                      numEpoch=100,
                      batchSize=20,
                      epsilonw=0.1)

        start = time.time() # inicia el temporizador
        #entrena la red
        dbn0.train(dataTrn=binaryTrnData,
                   dataVal=binaryValData,
                   path=fullPath)
        end = time.time()   # fin del temporizador
        print("Tiempo total: {}".format(timer(start,end)))

        dbn0.save(fullPath + "dbnMNIST", compression='zip')

    else: # no guardar, levanto de disco

        miDBN = dbn.load(filename=fullPath + "dbnMNIST", compression='zip')
        print(miDBN)

        red_capas = load_dbn_weigths(fullPath, miDBN.name)
        assert False

        # Dependencias Externas
        from cupydle.dnn.NeuralNetwork import NeuralNetwork
        from cupydle.dnn.NeuralNetwork import load as netLoad
        from cupydle.test.mnist.mnist import MNIST
        from cupydle.test.mnist.mnist import open4disk
        from cupydle.test.mnist.mnist import save2disk

        def label_to_vector(label, n_classes):
            lab = numpy.zeros((n_classes, 1), dtype=numpy.int8)
            label = int(label)
            lab[label] = 1
            return numpy.array(lab).transpose()


        # se leen de disco la base de datos
        mn = open4disk(filename=dataPath + setName, compression='bzip2')
        #mn.info                                        # muestra alguna informacion de la base de datos

        # obtengo todos los subconjuntos
        train_img,  train_labels= mn.get_training()
        test_img,   test_labels = mn.get_testing()
        val_img,    val_labels  = mn.get_validation()

        # umbral para la binarizacion
        threshold = 128
        binaryTrnData = (train_img>threshold).astype(numpy.float32)
        binaryValData = (val_img>threshold).astype(numpy.float32)
        binaryTstData = (test_img>threshold).astype(numpy.float32)

        y_tr = [label_to_vector(y, 10)[0] for y in train_labels]
        training_data = [(x, y) for x, y in zip(binaryTrnData, y_tr)]

        y_val = [label_to_vector(y, 10)[0] for y in val_labels]
        validation_data = [(x, y) for x, y in zip(binaryValData, y_val)]

        y_tst = numpy.reshape(numpy.array(test_labels, dtype=float), (len(test_labels), 1))
        testing_data = [(x, y) for x, y in zip(binaryTstData, y_tst)]
        #testing_data = [(x, y) for x, y in zip(binaryTstData, test_labels)]


        # cargo los pesos solamente de la red, ojo que deben estar transpuestos
        # (salida, entrada) por la nomeclatura adoptada en los multilayer perceptrons
        lista_pesos = [rbm.w.get_value().transpose() for rbm in red_capas]

        # parametros de la red
        n_visible = 784
        batchSize = 50
        capas_tmp = [pesos.shape[0] for pesos in lista_pesos]
        capas = [n_visible]
        capas.extend(capas_tmp)
        net = NeuralNetwork(list_layers=capas,
                            clasificacion=True,
                            funcion_error="MSE", ##"CROSS_ENTROPY" / "MSE"
                            funcion_activacion="Sigmoid",
                            w=lista_pesos,
                            b=None)

        print("Training Data size: " + str(len(training_data)))
        print("--trn:", "x:",training_data[0][0].shape, "y:", training_data[0][1].shape)
        print("Validating Data size: " + str(len(validation_data)))
        print("--val:", "x:",validation_data[0][0].shape, "y:", validation_data[0][1].shape)
        print("Testing Data size: " + str(len(testing_data)))
        print("--tst:", "x:",testing_data[0][0].shape, "y:", testing_data[0][1].shape)

        training_data2 =    [(x, y) for x, y in zip(binaryTrnData, numpy.reshape(numpy.array(train_labels, dtype=float), (len(train_labels), 1)))]
        validation_data2 =  [(x, y) for x, y in zip(binaryValData, numpy.reshape(numpy.array(val_labels, dtype=float), (len(val_labels), 1)))]
        testing_data2 =     [(x, y) for x, y in zip(binaryTstData, numpy.reshape(numpy.array(test_labels, dtype=float), (len(test_labels), 1)))]

        import random

        print("Entrenando...")
        net.fit(train=training_data2,
                valid=validation_data2,
                test=testing_data2,
                batch_size=batchSize,
                epocas=100,
                tasa_apren=0.2,
                momentum=0.1)
        net.save(fullPath + miDBN.name + '_Train')


        net = netLoad(fullPath + miDBN.name + '_Train')
        for i in range(0,10):
            indice = random.randint(0,len(testing_data2))
            prediccion = net.predict(testing_data2[indice][0])
            real = testing_data2[indice][1]
            print("indice: {} \t Real: {} \t Prediccion: {}".format(indice, real, prediccion))

        """
        optirun python3 cupydle/dnn/dbn_gpu.py

        Using gpu device 0: GeForce GT 420M (CNMeM is disabled, cuDNN not available)
        El archivo mnist en /run/media/lerker/Documentos/Proyecto/Codigo/cupydle/cupydle/data/DB_mnist/ ya existe, saliendo...
        Name: dbnTest
        Cantidad de capas: 3
        -[1] :
        Numero de neuronas visibles: 784
        Numero de neuronas ocultas: 500
        Numero de epocas: 15
        Tasa de aprendizaje para los pesos: 0.1
        Tasa de aprendizaje para las unidades visibles None
        Tasa de aprendizaje para las unidades ocultas: None
        Castigo pesos: None
        Tasa de momento inicial: None
        Tasa de momento final: None
        Tamanio del batch: 50

        -[2] :
        Numero de neuronas visibles: 500
        Numero de neuronas ocultas: 100
        Numero de epocas: 15
        Tasa de aprendizaje para los pesos: 0.1
        Tasa de aprendizaje para las unidades visibles None
        Tasa de aprendizaje para las unidades ocultas: None
        Castigo pesos: None
        Tasa de momento inicial: None
        Tasa de momento final: None
        Tamanio del batch: 50

        -[3] :
        Numero de neuronas visibles: 100
        Numero de neuronas ocultas: 10
        Numero de epocas: 15
        Tasa de aprendizaje para los pesos: 0.1
        Tasa de aprendizaje para las unidades visibles None
        Tasa de aprendizaje para las unidades ocultas: None
        Castigo pesos: None
        Tasa de momento inicial: None
        Tasa de momento final: None
        Tamanio del batch: 50


        /run/media/lerker/Documentos/Proyecto/Codigo/cupydle/cupydle/test/mnist/test1/dbnTest_capa1.pgz
        /run/media/lerker/Documentos/Proyecto/Codigo/cupydle/cupydle/test/mnist/test1/dbnTest_capa2.pgz
        /run/media/lerker/Documentos/Proyecto/Codigo/cupydle/cupydle/test/mnist/test1/dbnTest_capa3.pgz
        <cupydle.dnn.rbm_gpu.rbm_gpu object at 0x7f7250067e48>
        Training Data size: 50000
        --trn: x: (784,) y: (10,)
        Validating Data size: 10000
        --val: x: (784,) y: (10,)
        Testing Data size: 10000
        --tst: x: (784,) y: (1,)
        Entrenando...
        Epoch 0 training complete - Error: 84.03 [%]- Tiempo: 154.4885 [s]
        Epoch 1 training complete - Error: 83.8 [%]- Tiempo: 137.5257 [s]
        Tiempo total requerido: 292.0166 [s]
        Final Score 16.27
        indice: 541      Real: [ 4.]     Prediccion: 6
        indice: 9613     Real: [ 2.]     Prediccion: 6
        indice: 3570     Real: [ 5.]     Prediccion: 1
        indice: 9063     Real: [ 3.]     Prediccion: 3
        indice: 5504     Real: [ 3.]     Prediccion: 0
        indice: 7898     Real: [ 0.]     Prediccion: 9
        indice: 3994     Real: [ 5.]     Prediccion: 8
        indice: 8997     Real: [ 6.]     Prediccion: 9
        indice: 1685     Real: [ 6.]     Prediccion: 9
        indice: 777      Real: [ 1.]     Prediccion: 7

        """
