#!/usr/bin/env python3

# TODO
# explota por falta de memoria, se ve que algo que hice del free energy

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



class dbn(object):

    def __init__(self, numpy_rng=None, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 100], n_outs=None, name=None):
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

        # hidden units
        self.hidden_layers_sizes = [500,100]

        self.n_layers = len(self.hidden_layers_sizes)

        assert self.n_layers > 0, "No hay nada que computar"

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

        self.weights = []
        self.layers = []

        self.n_ins = n_ins

        if name is None:
            name="dbnTest"
        self.name = name


    def train(self, dataTrn, dataVal, path):
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        self.x = dataTrn

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = self.n_ins
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layers[-1]


            # Construct an RBM that shared weights with this layer
            rbm_layer = rbm_gpu(n_visible=input_size, n_hidden=self.hidden_layers_sizes[i])

            # train it!! layer per layer
            print("Entrenando la capa:", i+1)
            rbm_layer.maxEpoch = 15
            rbm_layer.train(data=layer_input,
                            miniBatchSize=50,
                            validationData=dataVal)

            print("Guardando los pesos..")
            filename = path + self.name + "_capa" + str(i+1) + ".pgz"
            rbm_layer.save(filename, absolutName=True)

            # ahora debo tener las entras que son las salidas del modelo anterior (activaciones de las ocultas)
            # TODO aca debo tener las activaciones de las ocultas
            [_, hiddenActPos, _, _] = rbm_layer.sampler(layer_input)
            [_, dataVal, _, _] = rbm_layer.sampler(dataVal)

            print("Guardando las muestras para la siguiente capa..")
            filename_pesos = path + self.name + "_capaPesos" + str(i+1)
            import pickle
            import gzip # gzip
            with gzip.GzipFile(filename_pesos + '.pgz', 'w') as f:
                pickle.dump(hiddenActPos, f)
                f.close()
            self.layers.append(hiddenActPos)
            del rbm_layer
        # FIN FOR

        print("Terminado")
        return 1
    # FIN TRAIN



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

    os.chdir(currentPath+'/cupydle/data/')                  # me muevo al directorio de los datos
    import subprocess
    subprocess.call(testPath + 'get_data.sh', shell=True)   # chequeo si necesito descargar los datos
    os.chdir(currentPath)                                   # vuelvo al directorio original

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

    if guardar:
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

        # umbral para la binarizacion
        threshold = 128

        # parametros de la red
        n_visible = 784
        batchSize = 50

        binaryTrnData = (train_img>threshold).astype(numpy.float32)
        binaryValData = (val_img>threshold).astype(numpy.float32)

        mi_dbn = dbn(n_ins=n_visible, hidden_layers_sizes=[500,100], name=None)


        start = time.time() # inicia el temporizador

        #entrena la red
        mi_dbn.train(dataTrn=binaryTrnData,
                     dataVal=binaryValData,
                     path=fullPath)
        end = time.time()   # fin del temporizador
        print("Tiempo total: {}".format(timer(start,end)))

    else:
        red = rbm_gpu.load(fullPath + 'dbnTest_capa1.pgz')
        w1=red.w.get_value()
        del red

        # Dependencias Externas
        from cupydle.dnn.NeuralNetwork import NeuralNetwork
        from cupydle.test.mnist.mnist import MNIST
        from cupydle.test.mnist.mnist import open4disk
        from cupydle.test.mnist.mnist import save2disk

        def label_to_vector(label, n_classes):
            lab = numpy.zeros((n_classes, 1), dtype=numpy.int8)
            label = int(label)
            lab[label] = 1
            return numpy.array(lab).transpose()

        # Seteo de los parametros
        capa_entrada    = 784
        capa_oculta_1   = 500
        capa_salida     = 10
        capas           = [capa_entrada, capa_oculta_1, capa_salida]


        # se leen de disco los datos
        mn = open4disk(filename=dataPath + setName, compression='bzip2')
        #mn.info                                        # muestra alguna informacion de la base de datos

        # obtengo todos los subconjuntos
        train_img,  train_labels= mn.get_training()
        test_img,   test_labels = mn.get_testing()
        val_img,    val_labels  = mn.get_validation()

        # parametros de la red
        n_visible = 784
        batchSize = 50

        # umbral para la binarizacion
        threshold = 128
        binaryTrnData = (train_img>threshold).astype(numpy.float32)
        binaryValData = (val_img>threshold).astype(numpy.float32)
        binaryTstData = (test_img>threshold).astype(numpy.float32)

        y_tr = [label_to_vector(y, 10)[0] for y in train_labels]
        training_data = [(x, y) for x, y in zip(binaryTrnData, y_tr)]




        print(w1.shape) #784,500
        # w1 W[(500, 784)]:
        # w2 W[(10, 500)]:
        lista_pesos = []
        lista_pesos.append(w1.transpose())

        # esto es de la ultima capa, la cual no deberia importarme, solo se inicializan las primeras, esta es de clasificacion
        w_tmp = numpy.asarray(
                numpy.random.uniform(low=-numpy.sqrt(6.0 / (500 + 10)),
                                  high=+numpy.sqrt(6.0 / (500 + 10)),
                                  size=(500,10)),
                dtype=numpy.dtype(float))
        lista_pesos.append(w_tmp.transpose())

        net = NeuralNetwork(list_layers=capas, clasificacion=True, funcion_error="CROSS_ENTROPY",
                            funcion_activacion="Sigmoid", w=lista_pesos, b=None)

        y_val = [label_to_vector(y, 10)[0] for y in val_labels]
        validation_data = [(x, y) for x, y in zip(binaryValData, y_val)]

        y_tst = numpy.reshape(numpy.array(test_labels, dtype=float), (len(test_labels), 1))
        testing_data = [(x, y) for x, y in zip(binaryTstData, y_tst)]
        #testing_data = [(x, y) for x, y in zip(binaryTstData, test_labels)]


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
        """
        for i in range(0,10):
            indice = random.randint(0,len(testing_data2))
            prediccion = net.predict(testing_data2[indice][0])
            real = testing_data2[indice][1]
            print("indice: {} \t Real: {} \t Prediccion: {}".format(indice, real, prediccion))

        #
        """
        print("Entrenando...")
        net.fit(train=training_data2, valid=validation_data2, test=testing_data2, batch_size=50, epocas=25, tasa_apren=0.2, momentum=0.1)
        """

        for i in range(0,10):
            indice = random.randint(0,len(testing_data2))
            prediccion = net.predict(testing_data2[indice][0])
            real = testing_data2[indice][1]
            print("indice: {} \t Real: {} \t Prediccion: {}".format(indice, real, prediccion))
        net.save(fullPath + 'mnist_demo_dbn')
        """
        from cupydle.dnn.NeuralNetwork import load as Nload
        net = Nload(fullPath + 'mnist_demo_dbn.cupydle')
        for i in range(0,10):
            indice = random.randint(0,len(testing_data2))
            prediccion = net.predict(testing_data2[indice][0])
            real = testing_data2[indice][1]
            print("indice: {} \t Real: {} \t Prediccion: {}".format(indice, real, prediccion))
        """
        Epoch 0 training complete - Error: 76.41 [%]- Tiempo: 105.3323 [s]
        Epoch 1 training complete - Error: 73.7 [%]- Tiempo: 114.0037 [s]
        Epoch 2 training complete - Error: 74.6 [%]- Tiempo: 113.7274 [s]
        Epoch 3 training complete - Error: 49.32 [%]- Tiempo: 125.6869 [s]
        Epoch 4 training complete - Error: 46.4 [%]- Tiempo: 117.2031 [s]
        Epoch 5 training complete - Error: 47.78 [%]- Tiempo: 112.594 [s]
        Epoch 6 training complete - Error: 42.86 [%]- Tiempo: 126.6591 [s]
        Epoch 7 training complete - Error: 36.1 [%]- Tiempo: 147.8314 [s]
        Epoch 8 training complete - Error: 36.14 [%]- Tiempo: 149.3755 [s]
        Epoch 9 training complete - Error: 34.47 [%]- Tiempo: 136.7185 [s]
        Epoch 10 training complete - Error: 30.33 [%]- Tiempo: 141.6036 [s]
        """
