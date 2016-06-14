#!/usr/bin/python3
import numpy

from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk

# para ejecutar programas
from subprocess import call

import time
import sys




class rbm(object):

    def __init__(self, n_visible, n_hidden=1000):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param w: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """
        self.epsilonw = 0.1  # Learning rate for weights
        self.epsilonvb = 0.1  # Learning rate for biases of visible units
        self.epsilonhb = 0.1  # Learning rate for biases of hidden units
        self.weightcost = 0.0002
        self.initialmomentum = 0.5
        self.finalmomentum = 0.9

        self.momentum = 0.6

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.evolution = []

        # create a number generator
        numpy_rng = numpy.random.RandomState(1234)


        # W is initialized with `initial_W` which is uniformely
        # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
        # converted using asarray to dtype theano.config.floatX so
        # that the code is runable on GPU
        self.w = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    size=(self.n_visible, self.n_hidden)
                ),
                dtype=numpy.float32
            )


        ########
        # bias de las unidades ocultas    b
        self.hidbiases  = numpy.zeros(shape=(1, self.n_hidden), dtype=numpy.float32)
        # bias de las unidades visibles   a
        self.visbiases  = numpy.zeros(shape=(1, self.n_visible), dtype=numpy.float32)

        # Todo debe ser un parametro externo
        self.maxepoch = 15
        self.numcases = 100  # los numeros de casos son la cantidad de patrones en el bacth (filas)

        self.alEstiloHinton=True
    # END INIT

    def contrastiveDivergence(self, conjunto):
        # cantidad de patrones del mini-batch
        self.numcases = conjunto.shape[0]

        self.numcases = 100

        # positivo: probabilidades de las unidades ocultas
        poshidprobs = numpy.zeros(shape=(self.numcases,self.n_hidden), dtype=numpy.float32) # (2,1000)
        # negativo: probabilidades de las unidades ocultas
        neghidprobs = numpy.zeros(shape=(self.numcases,self.n_hidden), dtype=numpy.float32) # (2,1000)
        # positivo: poductos
        posprods    = numpy.zeros(shape=(self.n_visible,self.n_hidden), dtype=numpy.float32) # (784,1000)
        # negativo: productos
        negprods    = numpy.zeros(shape=(self.n_visible,self.n_hidden), dtype=numpy.float32) # (784,1000)
        # incremento de pesos    delta_W
        vishidinc   = numpy.zeros(shape=(self.n_visible, self.n_hidden), dtype=numpy.float32) # (784,1000)
        # incrementos de los bias ocultos
        hidbiasinc  = numpy.zeros(shape=(1,self.n_hidden), dtype=numpy.float32) # (1, 1000)
        # incrementos de los bias visibles
        visbiasinc  = numpy.zeros(shape=(1,self.n_visible), dtype=numpy.float32) # (1,784)

        maxEpoch = 10

        indices = range(0, conjunto.shape[0], 1)
        batchSize = 100
        idx = [(x,y) for x,y in zip(indices[0:-batchSize:batchSize], indices[batchSize:-1:batchSize])]


        print("Training Dataset size: \t{}".format(conjunto.shape[0]))
        print("Visible units: \t\t{}".format(self.n_visible))
        print("Hidden units: \t\t{}".format(self.n_hidden))
        print("Batch size: \t\t{}".format(self.numcases))
        print("Maximum Epoch: \t\t{}".format(self.maxepoch))
        print("-------------------------------------------------------")

        if self.alEstiloHinton:
            print("Metodo Hinton")
        else:
            print("Metodo Estandar")

        #para cada epoca hasta el numero maximo de epocas
        for epoch in range(0, maxEpoch):
            print('Starting Epoch {} of {}'.format(epoch, maxEpoch))
            errorSum = 0.0

            # para cada mini batch, opera con todos los patrones del minibatch
            for i in range(0, len(idx)):
                miniBatch = conjunto[idx[i][0]: idx[i][1]: 1]
                self.numcases = miniBatch.shape[0]

                if self.alEstiloHinton:
                    # todo este if puede desaparecer, hinton mejoro el calculo de CD y actualizacion de pesos

                    # aca como lo copie de hinton
                    poshidprobs, poshidstates, posprods, poshidact, posvisact = self.positive(miniBatch)
                    negdata, neghidprobs, negprods, neghidact, negvisact = self.negative(poshidstates)
                    # esta es la misma que la anterior pero todo en una
                    ###negdata = self.CD_1_hinton(miniBatch)

                    vishidinc, visbiasinc, hidbiasinc = self.updateParams_hinton(posprods=posprods,
                                                                                negprods=negprods,
                                                                                posvisact=posvisact,
                                                                                negvisact=negvisact,
                                                                                poshidact=poshidact,
                                                                                neghidact=neghidact,
                                                                                vishidinc=vishidinc,
                                                                                visbiasinc=visbiasinc,
                                                                                hidbiasinc=hidbiasinc,
                                                                                epoch=epoch,
                                                                                maxEpoch=maxEpoch)
                    err = numpy.sum(a=numpy.sum(numpy.power((miniBatch-negdata),2), dtype=numpy.float32), dtype=numpy.float32)
                else:
                    # es es como hace michaela y mepa que es asi
                    #visibleRec, hiddenAct, hiddenRec = self.contrastiveDivergence_1step(miniBatch)
                    visibleRec, hiddenAct, hiddenRec = self.contrastiveDivergence_kstep(miniBatch,3)
                    vishidinc, visbiasinc, hidbiasinc = self.updateParams(  visible=miniBatch,
                                                                            hidden=hiddenAct,
                                                                            visibleRec=visibleRec,
                                                                            hiddenRec=hiddenRec,
                                                                            vishidinc=vishidinc,
                                                                            visbiasinc=visbiasinc,
                                                                            hidbiasinc=hidbiasinc,
                                                                            epoch=epoch,
                                                                            maxEpoch=maxEpoch)
                    err = numpy.sum(numpy.power((miniBatch-visibleRec),2))

                errorSum += err
                #err = numpy.sum(numpy.power((miniBatch-visibleRec),2))

                # fin for de cada minibatch

            print("Epoch {} - Error {}".format(epoch, errorSum))
            #end for epoch

        return
    # END contrastiveDivergence

    def updateParams(self, visible, hidden, visibleRec, hiddenRec, vishidinc, visbiasinc, hidbiasinc, epoch, maxEpoch):
        """
        """
        #print(visible.shape) # (100, 784)
        #print(visibleRec.shape) # (100, 784)
        #print(hidden.shape) # (100, 1000)
        #print(hiddenRec.shape) # (100, 1000)

        # TODO lo copio de hinton
        if epoch > maxEpoch/2:
            momentum = self.finalmomentum
        else:
            momentum = self.initialmomentum
        ##

        positiva = numpy.dot(visible.T, hidden) # (784, 1000)
        #print("Positiva:", positiva.shape)
        negativa = numpy.dot(visibleRec.T, hiddenRec) # (784, 1000)
        #print("Negativa:", negativa.shape)

        vishidinc = momentum * vishidinc + (self.epsilonw / self.numcases) * (positiva - negativa) - self.weightcost*self.w # delta pesos
        self.w    = self.w + vishidinc # actualizacion de los pesos
        #print("Pesos:", self.w.shape) # (784, 1000)

        visbiasinc     = momentum*visbiasinc + (self.epsilonvb / self.numcases) * numpy.sum(a=(visible - visibleRec), axis=0)
        self.visbiases = self.visbiases + visbiasinc
        #print("Bias Visbles:", self.visbiases.shape) # (1, 784)

        hidbiasinc = momentum*hidbiasinc + (self.epsilonhb / self.numcases) * numpy.sum(a=(hidden - hiddenRec), axis=0)
        self.hidbiases = self.hidbiases + hidbiasinc
        #print("Bias Ocultos:", self.hidbiases.shape)

        return vishidinc, visbiasinc, hidbiasinc

    def updateParams_hinton(self, posprods, negprods, posvisact, negvisact, poshidact, neghidact, vishidinc, visbiasinc, hidbiasinc, epoch, maxEpoch):

        if epoch > maxEpoch/2:
            momentum = self.finalmomentum
        else:
            momentum = self.initialmomentum

        vishidinc = momentum * vishidinc + (self.epsilonw / self.numcases) * (posprods - negprods) - self.weightcost*self.w # delta pesos
        self.w    = self.w + vishidinc # actualizacion de los pesos
        #print("Pesos:", self.w.shape) # (784, 1000)

        visbiasinc     = momentum*visbiasinc + (self.epsilonvb / self.numcases) * numpy.sum(a=(posvisact - negvisact), axis=0)
        self.visbiases = self.visbiases + visbiasinc
        #print("Bias Visbles:", self.visbiases.shape) # (1, 784)

        hidbiasinc = momentum*hidbiasinc + (self.epsilonvb / self.numcases) * numpy.sum(a=(poshidact - neghidact), axis=0)
        self.hidbiases = self.hidbiases + hidbiasinc
        #print("Bias Ocultos:", self.hidbiases.shape)

        return vishidinc, visbiasinc, hidbiasinc


    def contrastiveDivergence_1step(self, visibleSample):
        """
        param: visibleSample es el vector de entrada

        return:
            visibleRec reconstruccion del vector visible
            hiddenAct activaciones del vector de unidades ocultas pasada hacia arriba
            hiddenRec reconstruccion del vector de las ocultas dada la visible

        """
        # asi creo que se hace

        linearSum   = numpy.dot(visibleSample, self.w) + numpy.tile(self.hidbiases, (self.numcases, 1))
        hidProb     = 1.0 / (1.0 + numpy.exp(-linearSum))
        hidden      = numpy.random.binomial(n=1, p=hidProb, size=hidProb.shape) # sample hidden units

        #reconstruction
        linearSum   = numpy.dot(hidden, self.w.T) + numpy.tile(self.visbiases, (self.numcases, 1))
        visProb     = 1.0 / (1.0 + numpy.exp(-linearSum))
        visibleRec = numpy.random.binomial(n=1, p=visProb, size=visProb.shape) # sample visible units RECONSTRUCTION

        linearSum   = numpy.dot(visibleRec, self.w) + numpy.tile(self.hidbiases, (self.numcases, 1))
        visProbRec  = 1.0 / (1.0 + numpy.exp(-linearSum))
        hiddenRec   = numpy.random.binomial(n=1, p=visProbRec, size=visProbRec.shape) # sample hidden units

        hiddenAct = hidden
        return visibleRec, hiddenAct, hiddenRec

    def contrastiveDivergence_kstep(self, visibleSample, k):

        for i in range(0,k):

            linearSum   = numpy.dot(visibleSample, self.w) + numpy.tile(self.hidbiases, (self.numcases, 1))
            hidProb     = 1.0 / (1.0 + numpy.exp(-linearSum))
            hidden      = numpy.random.binomial(n=1, p=hidProb, size=hidProb.shape) # sample hidden units

            #reconstruction
            linearSum   = numpy.dot(hidden, self.w.T) + numpy.tile(self.visbiases, (self.numcases, 1))
            visProb     = 1.0 / (1.0 + numpy.exp(-linearSum))
            visibleRec = numpy.random.binomial(n=1, p=visProb, size=visProb.shape) # sample visible units RECONSTRUCTION

            if k>1:
                linearSum       = numpy.dot(hidden, self.w.T) + numpy.tile(self.visbiases, (self.numcases, 1))
                visProb         = 1.0 / (1.0 + numpy.exp(-linearSum))
                visibleSample   = numpy.random.binomial(n=1, p=visProb, size=visProb.shape) # sample visible units RECONSTRUCTION


        hiddenAct = hidden

        linearSum   = numpy.dot(visibleRec, self.w) + numpy.tile(self.hidbiases, (self.numcases, 1))
        visProbRec  = 1.0 / (1.0 + numpy.exp(-linearSum))
        hiddenRec   = numpy.random.binomial(n=1, p=visProbRec, size=visProbRec.shape) # sample hidden units

        return visibleRec, hiddenAct, hiddenRec

    def CD_1_hinton(self, data):
        poshidprobs = 1.0 / (1.0 + numpy.exp(numpy.dot(-data, self.w) -
                                             numpy.tile(self.hidbiases, (self.numcases, 1)) ))  # (2,1000)

        # TODO cambiar el random por uno que sea uniforme y no normal... pagina 5 hinton equ10
        poshidstates= poshidprobs > numpy.random.rand(self.numcases, self.n_hidden) # (2,1000)
        posprods    = numpy.dot(data.T, poshidprobs) # (784,1000)
        poshidact   = numpy.sum(a=poshidprobs, axis=0, dtype=numpy.float32)  # (1000,1)
        poshidact   = numpy.reshape(poshidact, (-1,self.n_hidden)) # para que coincida con hinton un vector fila # (1,1000)
        posvisact   = numpy.sum(a=data, axis=0, dtype=numpy.float32)  # (784,1)
        posvisact   = numpy.reshape(posvisact, (-1,self.n_visible)) # para que coincida con hinton un vector fila  # (1,784)
        negdata     = 1.0 / (1.0 + numpy.exp(numpy.dot(-poshidstates, self.w.T) -
                                         numpy.tile(self.visbiases, (self.numcases, 1)) )) #(2,1000)
        neghidprobs = 1.0 / (1.0 + numpy.exp(numpy.dot(-negdata, self.w) -
                                             numpy.tile(self.hidbiases, (self.numcases, 1)) ))
        negprods    = numpy.dot(negdata.T, neghidprobs)
        neghidact   = numpy.sum(a=neghidprobs, axis=0, dtype=numpy.float32)
        negvisact   = numpy.sum(negdata, axis=0, dtype=numpy.float32)

        return negdata


    def positive(self, data):
        """
        practicalHinton
        dado un patron elegido al azar, v. El estado binario de cada neurona oculta j, h_j
        es fijado a 1 con la probabilidad(7):
                 p(h_j=1|v) = sigma(b_j + sum_i(v_i w_ij))

                    sigma(x) = 1/(1+exp(-x))

        """
        # ecuacion (7) de practicalHinton
        # la multiplicacion es por batch...
        # data => (2, 784)... por fila hay un patron
        # w => (784,1000)
        # dot(data,w) => (2,1000)... por fila se tiene los patrones que fueron multiplicados por la matriz w.
        # tile(array,(l,p))... replica el array en una matriz de l filas por p columnas (de repeticion cada uno)
        # hay que restar a data*v-b... todo eso es la sigmodea
        poshidprobs = 1.0 / (1.0 + numpy.exp(numpy.dot(-data, self.w) -
                                             numpy.tile(self.hidbiases, (self.numcases, 1)) ))  # (2,1000)

        # la activacion de la neurona visible en la oculta determina el estado de estas ultimas
        # la neurona oculta se activa con valor 1 con la probabilidad antes calculada es mayor que un
        # numero random
        # TODO cambiar el random por uno que sea uniforme y no normal... pagina 5 hinton equ10
        poshidstates = poshidprobs > numpy.random.rand(self.numcases, self.n_hidden) # (2,1000)

        # tambien se calcula porque va a hacer necesario para despues lo siguiente
        posprods = numpy.dot(data.T, poshidprobs) # (784,1000)

        # suma todas las probabilidades por neurona a traves de sus activaciones con los diferentes patrones
        # lo hice asi para emular a hinton en matlab... axis=0 suma por columna, resultado es un arreglo fila
        # si axis es nula, suma todoooo
        poshidact = numpy.sum(a=poshidprobs, axis=0, dtype=numpy.float32)  # (1000,1)
        poshidact=numpy.reshape(poshidact, (-1,self.n_hidden)) # para que coincida con hinton un vector fila # (1,1000)
        # idem que el anterior
        posvisact = numpy.sum(a=data, axis=0, dtype=numpy.float32)  # (784,1)
        posvisact = numpy.reshape(posvisact, (-1,self.n_visible)) # para que coincida con hinton un vector fila  # (1,784)

        # TODO para mi que posprods es el resultado de la multiplicacion de data.T(v_i) con poshidstates (h_j) puede que este mal, no se porque hinton lo hizo asi
        return poshidprobs, poshidstates, posprods, poshidact, posvisact
    # END POSITIVE

    def negative(self, poshidstates):

        negdata = 1.0 / (1.0 + numpy.exp(numpy.dot(-poshidstates, self.w.T) -
                                         numpy.tile(self.visbiases, (self.numcases, 1)) )) #(2,1000)

        neghidprobs = 1.0 / (1.0 + numpy.exp(numpy.dot(-negdata, self.w) -
                                             numpy.tile(self.hidbiases, (self.numcases, 1)) ))


        negprods = numpy.dot(negdata.T, neghidprobs)


        neghidact = numpy.sum(a=neghidprobs, axis=0, dtype=numpy.float32)

        negvisact = numpy.sum(negdata, axis=0, dtype=numpy.float32)

        return negdata, neghidprobs, negprods, neghidact, negvisact
    # END NEGATIVE

    def save(self, filename=None, compression='gzip', layerN=0, absolutName=False):
        """
        guarda a disco la instancia de la RBM
        :param filename:
        :param compression:
        :param layerN: numero de capa a la cual pertence la rbm (DBN)
        :param absolutName: si es true se omite los parametros a excepto el filename
        :return:
        """
        from cupydle.dnn.utils import save as saver

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

        saver(object=self, filename=filename, compression=compression)
        return
    # END SAVE

    def load(self, filename=None, compression='gzip'):
        from cupydle.dnn.utils import load as loader
        try:
            self = loader(filename=filename, compression=compression)
        except:
            sys.stderr("Error al intentar abrir el archivo")
    # END LOAD

    @property
    def info(self):
        print("---------------- I N F O -------------------------")
        print("Numero de Unidades Visibles:", self.n_visible, self.visbiases.shape)
        print("Numero de Unidades Ocultas", self.n_hidden, self.hidbiases.shape)
        print("Matriz de pesos psinapticos:", self.w.shape)
        print("Tasa de aprendizaje para los pesos:", self.epsilonw)
        print("Tasa de aprendizaje para el bias visible:", self.epsilonvb)
        print("Tasa de aprendizaje para el bias oculto:", self.epsilonhb)
        print("regularizador del costo:", self.weightcost)
        print("Momento inicial:", self.initialmomentum)
        print("Momento final:", self.finalmomentum)
        print("Cantidad de epocas a entrenar:", self.maxepoch)
        print("Cantidad de patrones por batch:", self.numcases)
        print("--------------------------------------------------")

        return
    # END INFO


def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
# END TIMER

if __name__ == "__main__":

    """
    # ejecuto el script bash para descargar los datos
    call(["cupydle/data/get_mnist_data.sh"])
    # se crea el objeto mnist para cargar los datos...
    mn = MNIST('cupydle/data')
    # se guardan en disco los datos procesados
    save2disk(mn, filename='cupydle/data/mnistDB/mnistSet', compression='bzip2')
    """
    # se leen de disco los datos
    mn = open4disk(filename='cupydle/data/mnistDB/mnistSet', compression='bzip2')

    # muestra alguna informacion de la base de datos
    #mn.info

    # obtengo todos los subconjuntos
    train_img, train_labels = mn.get_training()
    test_img, test_labels = mn.get_testing()
    val_img, val_labels = mn.get_validation()

    # tomo solo un patron del conjunto para procesar (el primero por ej)
    #patron = mn.get_training()[0][0,:]
    patron1 = train_img[0]
    etiqueta1 = train_labels[0]
    patron2 = train_img[1]
    etiqueta2 = train_labels[1]
    #mn.plot_one_digit(patron1, label=etiqueta1)

    # umbral para la binarizacion
    threshold = 0
    patron_binario1 = patron1 > threshold
    patron_binario1.astype(int)
    patron_binario2 = patron2 > threshold
    patron_binario2.astype(int)
    #mn.plot_one_digit(patron_binario1, label=etiqueta)

    patron_binario1 = numpy.reshape(patron_binario1, (1,-1))
    patron_binario2 = numpy.reshape(patron_binario2, (1,-1))
    patron_binario = numpy.concatenate((patron_binario1,patron_binario2), axis=0)

    n_visible = 784
    red = rbm(n_visible)

    # mando a correr el algoritmo con un bach de 2 patrones de 784 datos cada uno
    #red.contrastiveDivergence(patron_binario)
    #shape patron_binario => (2,784)

    red.info

    start = time.time()
    red.contrastiveDivergence((train_img[0:200]>threshold).astype(int))
    end = time.time()

    red.save(filename="capa1", layerN=1, absolutName=True)

    red.load(filename="capa1")

    print("Tiempo total: {}".format(timer(start,end)))
    #http://imonad.com/rbm/restricted-boltzmann-machine/
    #https://github.com/deeplearningais/CUV/tree/master/examples/rbm
    #http://www.utstat.toronto.edu/~rsalakhu/DBM.html
