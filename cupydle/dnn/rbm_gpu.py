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

from cupydle.dnn.activations import Sigmoid


"""




"""
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

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
        self.epsilonvb      = theano.shared(numpy.asarray(0.1,   dtype=theanoFloat), name='epsilonvb')# Learning rate for biases of visible units
        self.epsilonhb      = theano.shared(numpy.asarray(0.1,   dtype=theanoFloat), name='epsilonhb')# Learning rate for biases of hidden units
        self.weightcost     = theano.shared(numpy.asarray(0.0002,dtype=theanoFloat), name='weightcost')# Weigth punishment
        self.momentum       = theano.shared(numpy.asarray(0.6,   dtype=theanoFloat), name='momentum')  # Momentum rate for default
        self.initialmomentum= theano.shared(numpy.asarray(0.5,   dtype=theanoFloat), name='momentum0') # Initial Mementum
        self.finalmomentum  = theano.shared(numpy.asarray(0.8,   dtype=theanoFloat), name='momentum1') # Las momentum change
        self.numcases       = theano.shared(numpy.asarray(0,     dtype=theanoFloat), name='numcases')  # Count cases per batch (buffer only for future)
        self.maxEpoch       = 15 # Maximium epochs for the trainig, only buffer

        # for allocate statistics
        self.statistics      = {'errorTraining':[], 'errorValidation': [], 'energyValidation':[]}

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
            w = theano.shared(value=_w, name='w')
            del _w

        # create shared variable for hidden units bias
        if hidbiases is None:
            #_hidbiases = numpy.zeros(shape=(1, self.n_hidden),
            #                         dtype=theanoFloat)
            _hidbiases = numpy.zeros(shape=(self.n_hidden),
                                     dtype=theanoFloat)
            hidbiases = theano.shared(value=_hidbiases, name='hidbiases')
            #hidbiases = theano.shared(value=_hidbiases, broadcastable=(True,False), name='hidbiases')
            del _hidbiases


        # create shared variable for visible units bias
        if visbiases is None:
            #_visbiases = numpy.zeros(shape=(1, self.n_visible),
            #                         dtype=theanoFloat)
            _visbiases = numpy.zeros(shape=(self.n_visible),
                                     dtype=theanoFloat)
            visbiases = theano.shared(value=_visbiases, name='visbiases')
            #visbiases = theano.shared(value=_visbiases, broadcastable=(True,False), name='visbiases')
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
    # END INIT

    def setParams(self, numEpoch, epsilonw=None, epsilonvb=None,
                    epsilonhb=None, weightcost=None, initialmomentum=None,
                    finalmomentum=None, activationFunction=None):
        self.epsilonw.set_value(epsilonw) if epsilonw is not None else None
        self.epsilonvb.set_value(epsilonvb) if epsilonvb is not None else None
        self.epsilonhb.set_value(epsilonhb) if epsilonhb is not None else None
        self.weightcost.set_value(weightcost) if weightcost is not None else None
        self.initialmomentum.set_value(initialmomentum) if initialmomentum is not None else None
        self.finalmomentum.set_value(finalmomentum) if finalmomentum is not None else None
        self.maxEpoch=numEpoch # no puede ser none
        # TODO
        self.activationFunction = Sigmoid()

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
            # theano se da cuenta y no es necesario realizar un 'theano.tensor.tile' del
            # bias (dimension), ya que lo hace automaticamente
            linearSum       = theano.tensor.dot(vsample, wG) + hbiasG
            hiddenActData   = self.activationFunction.nonDeterminstic(linearSum)
            #negative
            linearSum       = theano.tensor.dot(hiddenActData, wG.T) + vbiasG
            visibleActRec   = self.activationFunction.nonDeterminstic(linearSum)
            return [visibleActRec, hiddenActData]

        # loop o scan, devuelve los tres valores de las funcion y un diccionario de 'actualizaciones'
        ( [visibleActRec,
           hiddenActData],
          updates) = theano.scan(   fn           = oneStep,
                                    outputs_info = [msamples, None],
                                    non_sequences= [self.w, self.visbiases, self.hidbiases],
                                    n_steps      = steps,
                                    strict       = True,
                                    name         = 'scan_oneStepCD2')

        # last step, dont sample, only necessary for parameters updates
        linearSum       = theano.tensor.dot(visibleActRec, self.w) + self.hidbiases
        hiddenActRec    = self.activationFunction.activationProbablity(linearSum)

        return ([visibleActRec, hiddenActData, hiddenActRec], updates)

    def reconstructer_Kstep(self, vsamples, steps):
        """
        iterate over RBM from visibles units to hidden ones and back to visibles (recontruction) K step
        if K is equal to one, dont iterate, its not necessary generate 'scan' function
        so manualy calculate the output

        :type vsamples: tensor matrix
        :param vsamples: node to examples

        :type steps: integer
        :param steps: how many steps perform, normaly 1 step
        """
        # si la iteracion es de un solo paso, evito generar el scan, ahorra un poco de tiempo
        if steps == 1:
            # positive
            linearSum    = theano.tensor.dot(vsamples, self.w) + self.hidbiases
            hiddenActPos = self.activationFunction.nonDeterminstic(linearSum)
            #negative
            linearSum    = theano.tensor.dot(hiddenActPos, self.w.T) + self.visbiases
            visibleActRec= self.activationFunction.nonDeterminstic(linearSum)

            errorRec     = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleActRec - vsamples, 2),axis=0), axis=0)

            # en este caso como no se crea el scan, no me interesa la tercer salida
            # para la compilation en la funcion, el parametro 'updates' es None por default
            retorno = visibleActRec, errorRec, None
        else:
            ([visibleActRec, _, _], updates) = self.markovChain_k(vsamples, steps) # ejecuto k iteraciones

            visibleActRec = visibleActRec[-1]

            errorRec = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleActRec - vsamples, 2),axis=0), axis=0)

            retorno = visibleActRec, errorRec, updates

        return retorno

    def reconstructer_Kstep_v2(self, vsamples, steps):
        """
        iterate over RBM from visibles units to hidden ones and back to visibles (recontruction) K step
        if K is equal to one, dont iterate, its not necessary generate 'scan' function
        so manualy calculate the output

        :type vsamples: tensor matrix
        :param vsamples: node to examples

        :type steps: integer
        :param steps: how many steps perform, normaly 1 step
        """
        # si la iteracion es de un solo paso, evito generar el scan, ahorra un poco de tiempo
        if steps == 1:
            # positive
            linearSum    = theano.tensor.dot(vsamples, self.w) + self.hidbiases
            hiddenActPos = self.activationFunction.nonDeterminstic(linearSum)
            #negative
            linearSum    = theano.tensor.dot(hiddenActPos, self.w.T) + self.visbiases
            visibleActNeg= self.activationFunction.nonDeterminstic(linearSum)

            errorRec     = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleActNeg - vsamples, 2),axis=0), axis=0)

            # en este caso como no se crea el scan, no me interesa la tercer salida
            # para la compilation en la funcion, el parametro 'updates' es None por default
            retorno = visibleActNeg, hiddenActPos, errorRec, None
        else:
            ([visibleActNeg, hiddenActPos, _], updates) = self.markovChain_k(vsamples, steps) # ejecuto k iteraciones

            visibleActNeg = visibleActNeg[-1]
            hiddenActPos = hiddenActPos[-1]

            errorRec = theano.tensor.sum(theano.tensor.sum(theano.tensor.pow(visibleActNeg - vsamples, 2),axis=0), axis=0)

            retorno = visibleActNeg, hiddenActPos, errorRec, updates

        return retorno

    def sampler(self, data):
        # version sin batch
        """

        """
        data = theano.shared(numpy.asarray(data,dtype=theanoFloat), name='samplerData')

        costSamples = theano.tensor.matrix('samplerSamples')
        pasos=1
        visibleActNeg, hiddenActPos, errorTotal, updates = self.reconstructer_Kstep_v2(data, pasos)

        # la energia es un vector
        freeEnergyRec = self.free_energy(data)
        freeEnergyMean = theano.tensor.mean(freeEnergyRec)

        muestreador = theano.function(  inputs=[],
                                        outputs=[visibleActNeg, hiddenActPos, errorTotal, freeEnergyMean],
                                        updates=updates,
                                        name='samplerFun')

        [visibleActNeg, hiddenActPos, errorTotal, freeEnergyMean] = muestreador()

        return visibleActNeg, hiddenActPos, errorTotal, freeEnergyMean

    def cost(self, conjunto='validation'):
        # version sin batch
        """

        """
        data = False
        if conjunto == 'training':
            data = self.sharedData
        elif conjunto == 'validation':
            data = self.sharedValidationData
        else:
            print("parametro 'conjunto' no corresponde")
            sys.exit(0)

        costSamples = theano.tensor.matrix('costSamples')
        pasos=1

        _, errorTotal, updatesCost = self.reconstructer_Kstep(data, pasos)

        # la energia es un vector
        freeEnergyRec = self.free_energy(data)
        #freeEnergyVis = self.free_energy(data.get_value())
        #freeEnergyMean = theano.tensor.mean(freeEnergyVis) - theano.tensor.mean(freeEnergyRec)
        freeEnergyMean = theano.tensor.mean(freeEnergyRec)

        return errorTotal, freeEnergyMean, updatesCost

    def cost2(self, conjunto='validation'):
        """
        Separo en miniBatch el conjunto de validacion, para no quedarme sin memoria
        mepa que le falta un ejemplo o algo asi si no es entero jjusto el miniBatchSIze
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
        pasos=1
        visibleActRec, errorRec, updatesRec = self.reconstructer_Kstep(valsamples, pasos)

        # la energia es un vector
        freeEnergyRec = self.free_energy(valsamples)
        #freeEnergyVis = self.free_energy(data.get_value())
        #freeEnergyMean = theano.tensor.mean(freeEnergyVis) - theano.tensor.mean(freeEnergyRec)
        freeEnergyMean = theano.tensor.mean(freeEnergyRec)

        reconstructFunction = theano.function(
            inputs  = [index],
            outputs = [errorRec,freeEnergyMean],
            updates = updatesRec,
            givens  ={valsamples: data[index * miniBatchSize: (index + 1) * miniBatchSize]})


        tmp = [reconstructFunction(i) for i in range(nrMiniBatches)]
        errorTotal = 0.0
        energiaTotal = 0.0
        for i in range(nrMiniBatches):
            errorTotal += tmp[i][0]
            energiaTotal += tmp[i][1]

        del tmp
        return errorTotal, energiaTotal

    def sample(self, data):
        """
        Sampleador de los datos, fluye data sobre la red (entrenada?) y muestrea
        su salida como datos regenerados y las activaciones ocultas asi como el error
        se diferecia de reconstructer debido a que esta diseñado para ejecutarse sobre
        cualquier conjunto de datos pasado, ademas se ejecuta por batchs
        Separo en miniBatch la data

        """
        # TODO creo que esta haciendo un ejemplo de mas,,,

        numeroCasos = data.shape[0]
        visibleRec  = numpy.zeros(shape=(numeroCasos,self.n_visible))

        data = theano.shared(numpy.asarray(data,dtype=theanoFloat), name='sampleData')

        miniBatchSize = 1000

        nrMiniBatches   = int(numeroCasos / miniBatchSize)    # cantidad de iteraciones enteras
        restoCasos      = numeroCasos % miniBatchSize         # cantidad de casos por evaluar

        # Symbolic variable for index
        index = theano.tensor.iscalar('indiceSamples')
        dataSamples = theano.tensor.fmatrix('dataSamples')
        pasos = 1
        visibleActRec, errorRec, updatesRec = self.reconstructer_Kstep(dataSamples, pasos)

        reconstructFunction1 = theano.function(
                            inputs  =[index],
                            outputs =[visibleActRec, errorRec],
                            updates = updatesRec,
                            givens  ={dataSamples: data[index * miniBatchSize: (index + 1) * miniBatchSize]})

        tmp = [reconstructFunction1(i) for i in range(nrMiniBatches)]
        errorTotal = 0.0
        for i in range(nrMiniBatches):
            visibleRec[i * miniBatchSize: (i + 1) * miniBatchSize] = tmp[i][0]
            errorTotal += tmp[i][1]
        del tmp

        if restoCasos != 0:
            pasos = 1
            visibleActRec2, errorRec2, updatesRec2 = self.reconstructer_Kstep(dataSamples, pasos)
            reconstructFunction2 = theano.function(
                            inputs  =[],
                            outputs =[visibleActRec2, errorRec2],
                            updates =updatesRec2,
                            givens  ={dataSamples: data[nrMiniBatches * miniBatchSize:]})
            tmp2 = reconstructFunction2()
            visibleRec[nrMiniBatches * miniBatchSize:] = tmp2[0]
            errorTotal += tmp2[1]

        return visibleRec, errorTotal

    def free_energy(self, vsample):
        ''' Function to compute the free energy '''
        wx_b = theano.tensor.dot(vsample, self.w) + self.hidbiases
        vbias_term = theano.tensor.dot(vsample, self.visbiases)
        hidden_term = theano.tensor.sum(theano.tensor.log(1 + theano.tensor.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def updateParams(self, msamples, visibleActRec, hiddenActData, hiddenActRec, updates, epoch):
        # TODO arreglar lo de epoch
            if theano.tensor.ge(epoch,self.maxEpoch/2):
                momentum = self.finalmomentum
            else:
                momentum = self.initialmomentum

            positiva   = theano.tensor.dot(msamples.T, hiddenActData)      # (100,784)'*(1,300)>(784,300)
            negativa   = theano.tensor.dot(visibleActRec.T, hiddenActRec)  # (100,784)'*(1,300)>(784,300)

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

    def updateStatics(self, errorTrn=None, errorVal=None, errorTst=None, freeEnergy=None):
            self.statistics['errorTraining'].append(errorTrn)
            self.statistics['errorValidation'].append(errorVal)
            self.statistics['energyValidation'].append(freeEnergy)
            return 0

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
        msamples = theano.tensor.matrix(name="msamples")   # root
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

        # Para calcular el error durante el entrenamiento (recibe el indice el batch)
        # Para calcular el costo real se debe ejecutar despues de la funcion 'trainer'
        pasos = 1
        visibleReconstructionT, errorReconstructionT, updatesCostT = self.reconstructer_Kstep(msamples,pasos)
        costoBatchTraining = theano.function(inputs =[miniBatchIndex], # recibe el indice del batch a iterar
                                            outputs =[visibleReconstructionT, errorReconstructionT],
                                            givens  ={msamples: self.sharedData[miniBatchIndex*miniBatchSize: (miniBatchIndex+1)*miniBatchSize:1]},
                                            updates = updatesCostT,
                                            name    ='rbm_BatchCostTraining')


        #otra para el costo, total, del error de reconstruccion y de la energia libre
        if validationData is not None:
            errorReconstruccion, freeEnergy, updatesCost = self.cost(conjunto='validation')
            costTotal = theano.function(inputs  = [],
                                        outputs = [errorReconstruccion, freeEnergy],
                                        updates = updatesCost)

        # cantidad de indices... para recorrer el set
        indexCount = int(data.shape[0]/miniBatchSize)

        # error temporal acumulativo
        _errorTrn = theano.shared(numpy.asarray(0.0, dtype=theanoFloat), name='_errorTrn')

        if validationData is not None:
            errorValidation, energiaValidation = costTotal()
        else:
            errorValidation = 0.0; energiaValidation = 0.0

        for epoch in range(0, self.maxEpoch):
            print('Starting Epoch {} of {}, errorTrn:{}, errorVal:{}, freeEnergy:{}'.format(epoch, self.maxEpoch, _errorTrn.get_value(), errorValidation, energiaValidation), end='\r')
            _errorTrn.set_value(0.0)
            energiaValidation = 0.0
            for p in range(0, indexCount):
                trainer(p,1)
                [_,er] = costoBatchTraining(p)
                _errorTrn.set_value(_errorTrn.get_value() + er)
            # END SET

            # errores sobre el set entero en la epoca i
            if validationData is not None:
                errorValidation, energiaValidation1 = costTotal()
                energiaValidation = energiaValidation - energiaValidation1

            # los agrego al diccionario de estadisticas
            self.updateStatics(errorTrn=_errorTrn.get_value(), errorVal=errorValidation, freeEnergy=energiaValidation)

            #self.statistics['errorTraining'].append(_errorTrn.get_value())
            #self.statistics['errorValidation'].append(errorValidation)
            #self.statistics['energyValidation'].append(energiaValidation)
        # END epcoh
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior

        return 1

    def params(self):
        parametros = [v for k, v in self.__dict__.items()]
        print(parametros)

        return parametros


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
        funcion que importa desde disco una rbm, lo importante es que
        si el parametro 'method' es 'simple', se levanta el archivo como cuarquier
        clase guardada con pickle, comprimida o no.
        Si el parametro 'method' es 'theano' levanta el archivo como si se hibiera
        guardado con las funciones de theano. ver
        # http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
        hay que hacer un par de modificaciones aun
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


    def tile_raster_images(self, X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
        """
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.

        This function is useful for visualizing datasets whose rows are images,
        and also columns of matrices for transforming those rows
        (such as the first layer of a neural net).

        :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
        :param X: a 2-D array in which every row is a flattened image.

        :type img_shape: tuple; (height, width)
        :param img_shape: the original shape of each image

        :type tile_shape: tuple; (rows, cols)
        :param tile_shape: the number of images to tile (rows, cols)

        :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats

        :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not


        :returns: array suitable for viewing as an image.
        (See:`Image.fromarray`.)
        :rtype: a 2-d array with same dtype as X.

        """

        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape = [0,0]
        # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                          in zip(img_shape, tile_shape, tile_spacing)]

        if isinstance(X, tuple):
          assert len(X) == 4
          # Create an output numpy ndarray to store the image
          if output_pixel_vals:
              out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
          else:
              out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

          #colors default to 0, alpha defaults to 1 (opaque)
          if output_pixel_vals:
              channel_defaults = [0, 0, 0, 255]
          else:
              channel_defaults = [0., 0., 0., 1.]

          for i in range(4):
              if X[i] is None:
                  # if channel is None, fill it with zeros of the correct
                  # dtype
                  out_array[:, :, i] = numpy.zeros(out_shape,
                          dtype='uint8' if output_pixel_vals else out_array.dtype
                          ) + channel_defaults[i]
              else:
                  # use a recurrent call to compute the channel and store it
                  # in the output
                  out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
          return out_array

        else:
          # if we are dealing with only one channel
          H, W = img_shape
          Hs, Ws = tile_spacing

          # generate a matrix to store the output
          out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


          for tile_row in range(tile_shape[0]):
              for tile_col in range(tile_shape[1]):
                  if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                      if scale_rows_to_unit_interval:
                          # if we should scale values to be between 0 and 1
                          # do this by calling the `scale_to_unit_interval`
                          # function
                          this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                      else:
                          this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                      # add the slice to the corresponding position in the
                      # output array
                      out_array[
                          tile_row * (H+Hs): tile_row * (H + Hs) + H,
                          tile_col * (W+Ws): tile_col * (W + Ws) + W
                          ] \
                          = this_img * (255 if output_pixel_vals else 1)
          return out_array




def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
# END TIMER

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

    #if not os.path.exists(dataPath):
        #print("Creando la base de datos en:", dataPath)
        #os.makedirs(dataPath)

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
        n_hidden  = 500
        batchSize = 50

        # creo la red
        red = rbm_gpu(n_visible=n_visible, n_hidden=n_hidden)

        start = time.time() # inicia el temporizador

        #entrena la red
        red.train(  data=(train_img>threshold).astype(numpy.float32),   # los datos los binarizo y convierto a float
                    miniBatchSize=batchSize,
                    validationData=(val_img>threshold).astype(numpy.float32))

        end = time.time()   # fin del temporizador
        print("Tiempo total: {}".format(timer(start,end)))


        print('Guardando el modelo en ...', fullPath + modelName)
        start = time.time()
        red.save(fullPath + modelName, absolutName=True)
        end = time.time()
        print("Tiempo total para guardar: {}".format(timer(start,end)))

        print("FIN")
        print("ERROR Entranamiento:",red.statistics['errorTraining'])
        print("ERROR Validacion:",red.statistics['errorValidation'])
        print("ENERGIA Validacion:",red.statistics['energyValidation'])

        red.plot_error(red.statistics['errorTraining'],   show=True)
        red.plot_error(red.statistics['errorValidation'], show=True)
        red.plot_error(red.statistics['energyValidation'],show=True)
        red.plot_weigth()


    else:
        print("Abriendo el modelo desde...", fullPath + modelName)

        start = time.time()
        red2 = rbm_gpu.load(fullPath + modelName)

        end = time.time()
        print("Tiempo total para abrir: {}".format(timer(start,end)))

        red2.plot_error(red2.statistics['errorValidation'], tipo='validation',  save=True, show=False, path=fullPath).show()
        red2.plot_error(red2.statistics['errorTraining'],   tipo='training',    save=True, show=False, path=fullPath).show()
        red2.plot_error(red2.statistics['energyValidation'],tipo='validation',  show=False).show()
        red2.plot_weigth(save=True)

        #red2.tile_raster_images(X=red2.w.get_value(), img_shape=(28,28), tile_shape=(10,10))
        from cupydle.test.mnist.mnist import MNIST

        ejemplos = red2.sharedData.get_value()[0:10]
        ejemplostotal = red2.sharedData.get_value()

        e = 0.0

        ejemplos = red2.sharedData.get_value()[0:26000]
        vis, err = red2.sample(ejemplos)
        e += err
        ejemplos = red2.sharedData.get_value()[26001:]
        vis, err = red2.sample(ejemplos)
        e += err

        print("llegue..")
        vis2, err2 = red2.sample(ejemplostotal)
        print("errrrooooo2:",e)
        print("errrrooooo22222:",err2)
        print("")


        #MNIST.plot_ten_digits(ejemplos,crop=False)
        #MNIST.plot_ten_digits(vis, crop=False)
        # como imprimo los acultos? MNIST.plot_ten_digits(hid, crop=False)
