#!/usr/bin/env python3

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


"""




"""


class ActivationFunction(object):
    # TODO ver esto de generacion de randoms cada vez
    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        if 'theanoGenerator' in odict:
            del odict['theanoGenerator']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)   # update attributes

    def __getinitargs__():
        return None


class Sigmoid(ActivationFunction):
    def __init__(self):
        #self.theanoGenerator = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=numpy.random.randint(1, 1000))
        # o la version  MRG31k3p
        self.theanoGenerator = RandomStreams(seed=numpy.random.randint(1, 1000))

    def activationProbablity(self, x):
        return theano.tensor.nnet.sigmoid(x)

    def sample(self, x):
        # TODO por lo visto la binomial no esta implementada en CUDA,
        # por lo tanto lo lleva a la GPU a los datos
        # luego calcula la binomial (la cual la trae a la CPU de nuevo los datos)
        # y por ultimo lleva de nuevo a la GPU los datos calculado
        ### SOLUCION
        # deberia calcularse los numeros binomiales ({0,1}) en la GPU sin usar RandomStreams.binomial
        # si se retorna val y al theano.tensor.nnet.sigmoid(x) se le agrega 'transfer('gpu')' de la
        # activationProbability en el grafo se da cuenta de la optimizacion
        ###$
        # http://deeplearning.net/software/theano/tutorial/examples.html#example-other-random
        # There are 2 other implementations based on MRG31k3p and CURAND.
        # The RandomStream only work on the CPU, MRG31k3p work on the CPU and GPU. CURAND only work on the GPU.
        val = self.activationProbablity(x)
        return self.theanoGenerator.binomial(size=val.shape, n=1, p=val, dtype=theanoFloat)

####
class rbm_gpu(object):
    """Restricted Boltzmann Machine on GP-GPU (RBM-GPU)  """

    def __init__(
                self,
                n_visible=784,  # cantidad de neuronas visibles
                n_hidden=500,   # cantidad de neuronas ocultas
                w=None,         # matriz de pesos si ya cuento con una
                visbiases=None, # vector de biases visibles (si lo hay)
                hidbiases=None, # vector de biases ocultos (si lo hay)
                numpy_rng=None, # seed para numpy random
                theano_rng=None):#seed para theano random
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

        # TODO  arreglar esto, por lo visto tiene un problema para la multiplicacion
        #       de los valores por matrices en la GPU
        #       http://deeplearning.net/software/theano/tutorial/examples.html#basictutexamples
        #       Theano shared variable broadcast pattern default to False for each dimensions.
        #       Shared variable size can change over time, so we can’t use the shape to find
        #       the broadcastable pattern. If you want a different pattern, just pass it as
        #       a parameter theano.shared(..., broadcastable=(True, False))
        #
        self.epsilonw       = theano.shared(numpy.asarray(0.1,   dtype=theanoFloat), name='epsilonw')  # Learning rate for weights
        self.epsilonvb      = theano.shared(numpy.asarray(0.1,   dtype=theanoFloat), name='episilonvb')# Learning rate for biases of visible units
        self.epsilonhb      = theano.shared(numpy.asarray(0.1,   dtype=theanoFloat), name='episilonhb')# Learning rate for biases of hidden units
        self.weightcost     = theano.shared(numpy.asarray(0.0002,dtype=theanoFloat), name='weightcost')# Weigth punishment
        self.momentum       = theano.shared(numpy.asarray(0.6,   dtype=theanoFloat), name='momentum')  # Momentum rate for default
        self.initialmomentum= theano.shared(numpy.asarray(0.5,   dtype=theanoFloat), name='momentum0') # Initial Mementum
        self.finalmomentum  = theano.shared(numpy.asarray(0.8,   dtype=theanoFloat), name='momentum1') # Las momentum change
        self.numcases       = theano.shared(numpy.asarray(0,     dtype=theanoFloat), name='numcases')  # Count cases per batch (buffer only for future)
        self.maxEpoch       = 5 # Maximium epochs for the trainig, only buffer

        # for allocate statics
        self.evolution      = {'errorTraining':[], 'errorValidation': [], 'energia':[]}

        # create a number generator (fixed) for test NUMPY
        if numpy_rng is None:
            # TODO cambiar a aleatorio
            numpy_rng = numpy.random.RandomState(1234)

        # create a number generator (fixed) for test THEANO
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            # TODO cambiar a aleatorio
            theano_rng = RandomStreams(seed=1234)

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
            w = theano.shared(value=_w, name='w')
            del _w

        # create shared variable for hidden units bias
        if hidbiases is None:
            _hidbiases = numpy.zeros(shape=(1, self.n_hidden),
                                     dtype=theanoFloat)
            hidbiases = theano.shared(value=_hidbiases, name='hidbiases')
            del _hidbiases


        # create shared variable for visible units bias
        if visbiases is None:
            _visbiases = numpy.zeros(shape=(1, self.n_visible),
                                     dtype=theanoFloat)
            visbiases = theano.shared(value=_visbiases, name='visbiases')
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
        self.hidbiasinc  = theano.shared(value=numpy.zeros(shape=(1,self.n_hidden), dtype=theanoFloat), name='hidbiasinc')
        self.visbiasinc  = theano.shared(value=numpy.zeros(shape=(1,self.n_visible), dtype=theanoFloat), name='visbiasinc')
    # END INIT

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

    def markovChain_k(self, msamples, steps):
        """
        Ejecuta una cadena de Markov de 'k=steps' pasos

        :param msamples: nodo que representa al miniBatch
        :param steps: nodo escalar de la cantidad de pasos

        La Cadena de Markov comienza del ejemplo visible, luego se samplea
        la unidad oculta, el siguiente paso es samplear las unidades visibles
        a partir de la oculta (reconstruccion)
        """
        # un paso de CD es V->H->V
        def oneStep(vsample, wG, vbiasG, hbiasG):
            # positive
            linearSum       = theano.dot(vsample, wG) + theano.tensor.tile(hbiasG, (int(self.numcases.get_value()),1), ndim=None)
            hiddenActData   = self.activationFunction.sample(linearSum)
            #negative
            linearSum       = theano.dot(hiddenActData, wG.T) + theano.tensor.tile(vbiasG, (int(self.numcases.get_value()),1), ndim=None)
            visibleActRec   = self.activationFunction.sample(linearSum)
            return [visibleActRec, hiddenActData]

        # loop o scan, devuelve los tres valores de las funcion y un diccionario de 'actualizaciones'
        ( [visibleActRec,
           hiddenActData],
          updates) = theano.scan(   fn           = oneStep,
                                    outputs_info = [msamples, None],
                                    non_sequences= [self.w, self.visbiases, self.hidbiases],
                                    n_steps      = steps,
                                    strict       = True,
                                    name         = 'scan_oneStepCD')

        # last step, dont sample, only necessary for parameters updates
        linearSum       = theano.dot(visibleActRec, self.w) + theano.tensor.tile(self.hidbiases, (int(self.numcases.get_value()),1), ndim=None)
        hiddenActRec    = self.activationFunction.activationProbablity(linearSum)

        #visibleActRec = visibleActRec[-1]; hiddenActData = hiddenActData[-1]
        ##

        return ([visibleActRec, hiddenActData, hiddenActRec], updates)

    def reconstructer(self, vsamples):
        """
        iterate over RBM from visibles units to hidden ones and back to visibles (recontruction) one step
        """
        #positive
        linearSum       = theano.dot(vsamples, self.w) + theano.tensor.tile(self.hidbiases, (int(self.numcases.get_value()),1), ndim=None)
        hiddenActData   = self.activationFunction.sample(linearSum)
        #negative
        linearSum       = theano.dot(hiddenActData, self.w.T) + theano.tensor.tile(self.visbiases, (int(self.numcases.get_value()),1), ndim=None)
        visibleReconstruction = self.activationFunction.sample(linearSum)

        errorReconstruction = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleReconstruction - vsamples, 2),axis=0), axis=0)

        return visibleReconstruction, errorReconstruction

    def reconstructer2(self, vsamples, numcas):
        """
        iterate over RBM from visibles units to hidden ones and back to visibles (recontruction) one step
        """
        #positive
        linearSum       = theano.dot(vsamples, self.w) + theano.tensor.tile(self.hidbiases, (numcas,1), ndim=None)
        hiddenActData   = self.activationFunction.sample(linearSum)
        #negative
        linearSum       = theano.dot(hiddenActData, self.w.T) + theano.tensor.tile(self.visbiases, (numcas,1), ndim=None)
        visibleReconstruction = self.activationFunction.sample(linearSum)

        errorReconstruction = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleReconstruction - vsamples, 2),axis=0), axis=0)

        return visibleReconstruction, errorReconstruction

    def cost(self, conjunto='validation'):
        """
        Separo en miniBatch el conjunto de validacion, para no quedarme sin memoria
        """
        data = False
        if conjunto == 'training':
            data = self.sharedData
        elif conjunto == 'validation':
            data = self.sharedValidationData
        else:
            print("parametro 'conjunto' no corresponde")
            sys.exit(0)

        miniBatchSize = 1000
        nrMiniBatches = int(int(data.get_value().shape[0]) / miniBatchSize)

        # Symbolic variable for index
        index = theano.tensor.iscalar()
        valsamples = theano.tensor.fmatrix('valsamples')

        _, error = self.reconstructer2(valsamples, miniBatchSize)

        reconstructFunction = theano.function(
            inputs=[index],
            outputs=error,
            givens={valsamples: data[index * miniBatchSize: (index + 1) * miniBatchSize]})

        errores = [reconstructFunction(i) for i in range(nrMiniBatches)]
        errorTotal = 0.0
        for i in range(nrMiniBatches):
            errorTotal += errores[i]

        return errorTotal


    def free_energy_tmp(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def updateParams(self, msamples, visibleActRec, hiddenActData, hiddenActRec, updates, epoch):
        # TODO arreglar lo de epoch
            if theano.tensor.ge(epoch,self.maxEpoch/2):
                momentum = self.finalmomentum
            else:
                momentum = self.initialmomentum

            positiva   = theano.dot(msamples.T, hiddenActData)      # (100,784)'*(1,300)>(784,300)
            negativa   = theano.dot(visibleActRec.T, hiddenActRec)  # (100,784)'*(1,300)>(784,300)

            # calculo de los terminos
            # TODO fijarse lo de los bias, aplica como la suma sobre una nuerona de todas las activaciones... esta bien?
            # TODO es asi? como lo comentado tenia menos error, corroborar que se guardan los incrementos,
            #       la diferencia esta en que en el primero los asigno y en el segundo utilizo el 'updates'. que supuestamente deberia hacer mejor el trabajo
            """
            self.vishidinc  = momentum * self.vishidinc  + (self.epsilonw / self.numcases)  * (positiva - negativa) - self.weightcost*self.w # delta pesos
            self.visbiasinc = momentum * self.visbiasinc + (self.epsilonvb / self.numcases) * theano.tensor.sum(input=(msamples - visibleActRec), axis=0)
            self.hidbiasinc = momentum * self.hidbiasinc + (self.epsilonhb / self.numcases) * theano.tensor.sum(input=(hiddenActData - hiddenActRec), axis=0)
            """
            _vishidinc  = momentum * self.vishidinc  + (self.epsilonw / self.numcases)  * (positiva - negativa) - self.weightcost*self.w # delta pesos
            _visbiasinc = momentum * self.visbiasinc + (self.epsilonvb / self.numcases) * theano.tensor.sum(input=(msamples - visibleActRec), axis=0)
            _hidbiasinc = momentum * self.hidbiasinc + (self.epsilonhb / self.numcases) * theano.tensor.sum(input=(hiddenActData - hiddenActRec), axis=0)

            # construyo las actualizaciones a las cuales debo agregar al scan_update
            # las actualizaciones sobre las shared variables las realizo aca recien
            updates.update({self.w:         self.w          + _vishidinc})
            updates.update({self.visbiases: self.visbiases  + _visbiasinc})
            updates.update({self.hidbiases: self.hidbiases  + _hidbiasinc})

            # eliminar esta linea si se quiere descomentar lo anterior
            updates.update({self.vishidinc:  _vishidinc})
            updates.update({self.visbiasinc: _visbiasinc})
            updates.update({self.hidbiasinc: _hidbiasinc})

            return ([self.vishidinc, self.visbiasinc, self.hidbiasinc], updates)

    def train(self, data, miniBatchSize=100, validationData=None):
        print("Training an RBM, with | {} | visibles neurons and | {} | hidden neurons".format(self.n_visible, self.n_hidden))
        print("Data set size for Restricted Boltzmann Machine", len(data))

        # convierto todos los datos a una variable shared de theano
        self.sharedData  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='TrainingData')

        # para la validacion
        if validationData is not None:
            self.sharedValidationData = theano.shared(numpy.asarray(a=validationData, dtype=theanoFloat), name='ValidationData')

        # cantidad de ejemplos por batch
        self.numcases.set_value(miniBatchSize)

        # Theano NODES
        msamples = theano.tensor.fmatrix(name="msamples")   # root
        msamplesTrain = theano.tensor.fmatrix(name="msamplesTrain")

        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # realizar la cadena de markov k veces
        (   [visibleActRec,
             hiddenActData,
             hiddenActRec],
            updates        ) = self.markovChain_k(msamples, steps)

        # los valores que me interesan son los ultimos

        visibleActRec = visibleActRec[-1]
        hiddenActData = hiddenActData[-1]
        hiddenActRec  = hiddenActRec[-1]

        ###
        # TODO no me gusta que epoch sea un shared,
        # en base a las iteraciones del CD se deben calcular las incrementos
        epoch = theano.shared(1)
        (   [vishidinc,
             visbiasinc,
             hidbiasinc],
            updates     ) = self.updateParams(  msamples     =msamples,
                                                visibleActRec=visibleActRec,
                                                hiddenActData=hiddenActData,
                                                hiddenActRec =hiddenActRec,
                                                updates      =updates,
                                                epoch        =epoch)

        ### el error viejo, (antes de actualizar los parametros entrenados)
        #, sirve porque a este punto tengo todo calculado y no necesito recalcular nada
        # pero tener en cuenta que para el verdadero error se debe correr una vez mas afuera una vez ya actualizado los parametros
        old_error = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleActRec - msamples, 2),axis=0), axis=0)
        # es un error del batch


        ####
        # funcion principal (unica)
        # corre los minibatchs
        # calcula y actualiza los parametros
        # retorna el costo de la red "(pero antes de que se actualicen los parametros!!!)"
        trainer = theano.function(  inputs =[miniBatchIndex, steps],
                                    #outputs=[vishidinc, visbiasinc, hidbiasinc], # quitar,-> [] si se quiere que no retorne, es al pedo por ahora
                                    outputs=old_error,
                                    updates=updates,
                                    givens ={msamples: self.sharedData[miniBatchIndex*miniBatchSize: (miniBatchIndex+1)*miniBatchSize:1]},
                                    name   ='rbm_trainer')



        # calcula el costo de la red en el estado que este para iterar dentro del for de bacth y epoch
        # PARA EL COSTO SOBRE EL CONJUNTO DE ENTRENAMIENTO
        # Es para separar si se ejecuto 'trainer' antes, se retorna el costo actual del batch
        # paso de las recosntrucciones
        (visibleReconstruction, errorReconstruction) = self.reconstructer(msamples)

        # recibe el indice del batch a iterar
        costoBatchTraining = theano.function(inputs =[miniBatchIndex],
                                            outputs =[visibleReconstruction, errorReconstruction],
                                            givens  ={msamples: self.sharedData[miniBatchIndex*miniBatchSize: (miniBatchIndex+1)*miniBatchSize:1]},
                                            name    ='rbm_BatchCostTraining')


        ## llamar dentro del for del tran por epoch y  iteracion
        # ejecuto
        # cantidad de indices... para recorrer el set
        indexCount = int(data.shape[0]/miniBatchSize)


        ## para que este funcione se debe ejecutar en su totatilida, reconstructur no sabe el tamaño para el tile
        (visibleReconstruction1, errorReconstruction1) = self.reconstructer(msamples)
        sampler = theano.function(  inputs=[msamples],
                                    outputs=[visibleReconstruction1, errorReconstruction1],
                                    name   ='rbm_sampler')


        _errorTrn       = theano.shared(numpy.asarray(0.0, dtype=theanoFloat), name='_errorTrn')
        errorTraining   = theano.shared(numpy.asarray(0.0, dtype=theanoFloat), name='errorTraining')
        errorValidation = theano.shared(numpy.asarray(0.0, dtype=theanoFloat), name='errorValidation')

        for epoch in range(0, self.maxEpoch):
            print('Starting Epoch {} of {}, errorTrn:{}, errorVal:{}'.format(epoch, self.maxEpoch, errorTraining.get_value(), errorValidation.get_value()), end='\r')
            _errorTrn.set_value(0.0)
            for p in range(0, indexCount):
                trainer(p,1)
                [_,er] = costoBatchTraining(p)
                _errorTrn.set_value(_errorTrn.get_value() + er)
            # END SET

            # errores sobre el set entero en la epoca i
            errorTraining.set_value(_errorTrn.get_value())
            errorValidation.set_value(self.cost(conjunto='validation'))
            # los agrego al diccionario de estadicticas
            self.evolution['errorTraining'].append(errorTraining.get_value())
            self.evolution['errorValidation'].append(errorValidation.get_value())
        # END epcoh
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior
        print("FIN")
        print("ERROR Entranamiento:",self.evolution['errorTraining'])
        print("ERROR Validacion:",self.evolution['errorValidation'])

        return 1





def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
# END TIMER

if __name__ == "__main__":
    # para la prueba...
    from cupydle.test.mnist.mnist import MNIST
    from cupydle.test.mnist.mnist import open4disk
    from cupydle.test.mnist.mnist import save2disk

 # se leen de disco los datos
    mn = open4disk(filename='cupydle/data/mnistDB/mnistSet', compression='bzip2')

    # muestra alguna informacion de la base de datos
    #mn.info

    # obtengo todos los subconjuntos
    train_img, train_labels = mn.get_training()
    test_img, test_labels = mn.get_testing()
    val_img, val_labels = mn.get_validation()

    # umbral para la binarizacion
    threshold = 128
    n_visible = 784

    red = rbm_gpu(n_visible=n_visible,n_hidden=500)

    start = time.time()
    red.train(  data=(train_img>threshold).astype(numpy.float32),
                miniBatchSize=100,
                validationData=(val_img>threshold).astype(numpy.float32))
    end = time.time()

    print("Tiempo total: {}".format(timer(start,end)))

    red.plot_error(red.evolution['errorTraining'],show=False)
    red.plot_error(red.evolution['errorValidation'],show=True)
    red.plot_error(red.evolution['errorTraining'],tipo='training', show=False).show()

    red.plot_error(red.evolution['errorTraining'],tipo='training', show=False, save=True).show()

    """
    Using gpu device 0: GeForce GT 420M (CNMeM is disabled, cuDNN not available)
    RBM learningRate: epsilonw
    Data set size for Restricted Boltzmann Machine 50000
    FINrting Epoch 9 of 10, error:1509052.0
    ERROR: [array(2168984.0, dtype=float32), array(1913833.0, dtype=float32), array(1786226.0, dtype=float32), array(1697431.0, dtype=float32), array(1634844.0, dtype=float32), array(1600061.0, dtype=float32), array(1562577.0, dtype=float32), array(1525803.0, dtype=float32), array(1509052.0, dtype=float32), array(1491618.0, dtype=float32)]
    Tiempo total: 00:04:32.86

    ### cambiado los updates de los incrementos el comportamiento no es tan "suave"
    Using gpu device 0: GeForce GT 420M (CNMeM is disabled, cuDNN not available)
    RBM learningRate: epsilonw
    Data set size for Restricted Boltzmann Machine 50000
    FINrting Epoch 9 of 10, error:1516655.0
    ERROR: [array(2194691.0, dtype=float32), array(1897565.0, dtype=float32), array(1743270.0, dtype=float32), array(1623479.0, dtype=float32), array(1573964.0, dtype=float32), array(1583476.0, dtype=float32), array(1560282.0, dtype=float32), array(1544130.0, dtype=float32), array(1516655.0, dtype=float32), array(1530101.0, dtype=float32)]
    Tiempo total: 00:04:26.04


    #### calculando el error viejo.. osea en el mismo paso antes de actualizar lo que evita la multiplicacion delas matrices una vez mas, no continene el ultimo error, se reduce mucho el tiempo
    Using gpu device 0: GeForce GT 420M (CNMeM is disabled, cuDNN not available)
    Training an RBM, with | 784 | visibles neurons and | 500 | hidden neurons
    Data set size for Restricted Boltzmann Machine 50000
    FINrting Epoch 9 of 10, error:1513187.0
    ERROR: [array(2801426.0, dtype=float32), array(2018036.0, dtype=float32), array(1785388.0, dtype=float32), array(1668504.0, dtype=float32), array(1602905.0, dtype=float32), array(1570406.0, dtype=float32), array(1542211.0, dtype=float32), array(1535153.0, dtype=float32), array(1513187.0, dtype=float32), array(1505977.0, dtype=float32)]
    Tiempo total: 00:02:30.58


    """
