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

from cupydle.dnn.utils import temporizador

import numpy
import os
import shelve

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
from cupydle.dnn.utils import RestrictedDict


"""




"""
try:
    import PIL.Image as Image
except ImportError:
    import Image

# parte de las rbm
from cupydle.dnn.rbm import RBM
from cupydle.dnn.unidades import UnidadBinaria

# parte del ajuste fino
from cupydle.dnn.mlp import MLP

# utilidades
from cupydle.dnn.utils import save
from cupydle.dnn.utils import load as load_utils
from cupydle.dnn.utils import temporizador
import glob# load_dbn_weight
import matplotlib.pyplot
from cupydle.dnn.graficos import scale_to_unit_interval
from cupydle.dnn.graficos import imagenTiles


class rbmParams(object):
    # sirve para guardar el estado nomas
    def __init__(self, n_visible,
                n_hidden,
                epocas,
                tamMiniBatch,
                lr_pesos,
                pasosGibbs=1,
                w=None,
                lr_bvis=None,
                lr_bocu=None,
                costo_w=None,
                momento=0.0,
                toleranciaError=0.0,
                unidadesVisibles='binaria',
                unidadesOcultas='binaria'):
        self.n_visible          = n_visible
        self.n_hidden           = n_hidden
        self.epocas             = epocas
        self.lr_pesos           = lr_pesos
        self.pasosGibbs         = pasosGibbs
        self.w                  = w
        self.lr_bvis            = lr_bvis
        self.lr_bocu            = lr_bocu
        self.costo_w            = costo_w
        self.momento            = momento
        #self.toleranciaError    = toleranciaError
        self.tamMiniBatch       = tamMiniBatch
        self.unidadesVisibles   = unidadesVisibles
        self.unidadesOcultas    = unidadesOcultas
        return

    def __str__(self):
        print("Numero de neuronas visibles:",self.n_visible)
        print("Numero de neuronas ocultas:",self.n_hidden)
        print("Numero de epocas:",self.epocas)
        print("Tasa de aprendizaje para los pesos:",self.lr_pesos)
        print("Tasa de aprendizaje para las unidades visibles",self.lr_bvis)
        print("Tasa de aprendizaje para las unidades ocultas:",self.lr_bocu)
        print("Tasa de momento:",self.momento)
        print("Castigo pesos:",self.costo_w)
        print("Pasos de Gibss:", self.pasosGibbs)
        print("Tamanio del minibatch:",self.tamMiniBatch)
        #print("Tolerancia del error permitido:",self.toleranciaError)
        print("Tipo de Unidades Visibles:", self.unidadesVisibles)
        print("Tipo de Unidades Ocultas:", self.unidadesOcultas)
        return str("")

    @property
    def getParametrosEntrenamiento(self):

        tmpRBM = RBM().datosAlmacenar._allowed_keys
        retorno = {}
        for key in self.__dict__.keys():
            if key in tmpRBM:
                retorno[key] = self.__dict__[key]
        """
        for k in retorno:
            print(k, retorno[k])
        """
        del tmpRBM
        return retorno

class DBN(object):

    DEBUG = False

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

        # allocate symbolic variables for the data
        self.x = theano.tensor.matrix('samples')  # los datos de entrada son del tipo tensor de orden dos
        self.y = theano.tensor.ivector('labels')  # las etiquetas son tensores de orden uno (ints)

        self.capas = []
        self.n_layers = 0
        self.pesos = [] #pesos almacenados en una lista, una vez entrenados se guardan aca
                        # para utilizarse en el fit o prediccion...

        if name is None:
            name="dbnTest"
        self.name = name

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        self.ruta = ruta

        self.datosAlmacenar=self._initGuardar()

    def _initGuardar(self):
        """
        inicializa los campos para el almacenamiento
        """
        archivo = self.ruta + self.name + '.cupydle'

        # datos a guardar, es estatico, por lo que solo lo que se puede almacenar
        # debe setearse aca
        datos = {'tipo':                'dbn',
                 'nombre':              self.name,
                 'numpy_rng':           self.numpy_rng,
                 'pesos':               [],
                 'pesos_iniciales':     [],
                 'theano_rng':          self.theano_rng.rstate, # con set_value se actualiza
                 'activacionesOcultas': [],
                 'tasaAprendizaje':     0.0,
                 'regularizadorL1':     0.0,
                 'regularizadorL2':     0.0,
                 'momento':             0.0,
                 'epocas':              0.0,
                 'activacion':          'sigmoidea',
                 'toleranciaError':     0.0,
                 'costoTRN':            0.0,
                 'costoVAL':            0.0,
                 'costoTST':            0.0
                 }

        # diccionario restringido en sus keys
        almacenar = RestrictedDict(datos)
        for k,v in datos.items():
            almacenar[k]=v

        # se guarda a archivo
        # ojo con el writeback,
        # remuevo el archivo para comenzar desde cero
        #os.remove(archivo)
        # 'r'   Open existing database for reading only (default)
        # 'w' Open existing database for reading and writing
        # 'c' Open database for reading and writing, creating it if it doesn’t exist
        # 'n' Always create a new, empty database, open for reading and writing
        with shelve.open(archivo, flag='n', writeback=False) as shelf:
            for key in almacenar.keys():
                shelf[key] = almacenar[key]
            shelf.close()

        return almacenar

    def setParametros(self, parametros):
        self._guardar(diccionario=parametros)
        return 1

    def addLayer(self, n_visible=None, **kwargs):

        if n_visible is None:
            assert self.capas != [], "Debe prover de unidades de entrada para esta capa."
            n_visible = self.capas[-1].n_hidden

        # agrego una capa de rbm a la dbn, con los parametros que le paso
        self.capas.append(rbmParams(n_visible=n_visible, **kwargs))
        self.n_layers += 1

        return

    def entrenar(self, dataTrn, dataVal=None, pcd=True,
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

        T = temporizador()
        inicio = T.tic()

        for i in range(self.n_layers):
            # deben diferenciarse si estamos en presencia de la primer capa o de una intermedia
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self._cargar(key='activacionesOcultas')[-1] # al entrada es la anterior, la que ya se entreno

            # una carpeta para alojar los datos de la capa intermedia
            directorioRbm = self.ruta + 'rbm_capa{}/'.format(i+1)
            os.makedirs(directorioRbm) if not os.path.exists(directorioRbm) else None

            # Construct an RBM that shared weights with this layer
            capaRBM = RBM(n_visible=self.capas[i].n_visible,
                          n_hidden=self.capas[i].n_hidden,
                          w=self.capas[i].w,
                          ruta=directorioRbm)

            # configuro la capa, la rbm
            capaRBM.setParametros(self.capas[i].getParametrosEntrenamiento)

            # train it!! layer per layer
            print("Entrenando la capa:", i+1)
            self._guardar(diccionario={'pesos_iniciales':capaRBM.getW}) if guardarPesosIniciales else None

            capaRBM.entrenamiento(data=layer_input,
                                  tamMiniBatch=self.capas[i].tamMiniBatch,
                                  pcd=pcd,
                                  gibbsSteps=self.capas[i].pasosGibbs,
                                  validationData=dataVal,
                                  filtros=filtros)

            # en caso de debug guardo la capa entera
            if DBN.DEBUG:
                print("Guardando la capa..")
                filename = self.name + "_capa" + str(i+1)
                capaRBM.guardarObjeto(nombreArchivo=filename)

            # ahora debo tener las entras que son las salidas del modelo anterior (activaciones de las ocultas)
            # [probabilidad_H1, muestra_V1, muestra_H1]
            [_, _, hiddenProbAct] = capaRBM.muestra(muestraV=layer_input)

            # por ahora no lo utilizo
            if dataVal is not None:
                [_,_,dataVal] = capaRBM.muestra(muestraV=dataVal)

            # se guardan las activaciones ocultas para la siguiente iteracion
            # de la siguiente capa oculta
            print("Guardando las muestras para la siguiente capa..") if DBN.DEBUG else None
            self._guardar(diccionario={'activacionesOcultas':hiddenProbAct})

            # se guardan los pesos para el ajuste fino
            self._guardar(diccionario={'pesos':capaRBM.getW})

            del capaRBM
        # FIN FOR

        final = T.toc()
        print("Tiempo total para pre-entrenamiento DBN: {}".format(T.transcurrido(inicio, final)))

        return 0
    # FIN TRAIN

    def ajuste(self, datos, listaPesos=None, fnActivacion='sigmoidea',
                tambatch=10, semillaRandom=None):
        """
        construye un perceptron multicapa, y ajusta los pesos por medio de un
        entrenamiento supervisado.
        """
        if listaPesos is None:
            if self.pesos == []:
                print("Cargando los pesos almacenados...") if DBN.DEBUG else None
                self.pesos = self._cargar(key='pesos')
                if self.pesos==[]:
                    print("PRECAUCION!!!, esta ajustando la red sin entrenarla!!!") if DBN.DEBUG else None
                    self.pesos = [None] * self.n_layers
        else:
            self.pesos = listaPesos

        activaciones = []
        if fnActivacion is not None:
            if isinstance(fnActivacion, list):
                assert len(fnActivacion) == len(self.pesos), "No son suficientes funciones de activacion"
                activaciones = fnActivacion
            else:
                activaciones = [fnActivacion] * len(self.pesos)
        else:
            assert False, "Debe proveer una funcion de activacion"


        # se construye un mlp para el ajuste fino de los pesos con entrenamiento supervisado
        clasificador = MLP(clasificacion=True,
                           rng=semillaRandom,
                           ruta=self.ruta)

        ## solo cargo las keys del ajuste cargadas en las DBN al MLP buscando coincidencias y filtrando lo que no interesa
        tmp = clasificador.datosAlmacenar._allowed_keys
        retorno = {}
        for key in self.datosAlmacenar._allowed_keys:
            if key in tmp and key not in ['tipo', 'pesos', 'nombre', 'numpy_rng', 'theano_rng', 'pesos_iniciales']:
                retorno[key] = self._cargar(key=key)
        del tmp

        clasificador.setParametroEntrenamiento(retorno)
        del retorno

        # cargo en el perceptron multicapa los pesos en cada capa
        # como el fit es de clasificacion, las primeras n-1 capas son del tipo
        # 'logisticas' luego la ultima es un 'softmax'
        for i in range(0,len(self.pesos)-1):
            clasificador.agregarCapa(unidadesEntrada=self.capas[i].n_visible,
                                     unidadesSalida=self.capas[i].n_hidden,
                                     clasificacion=False,
                                     activacion=activaciones[i],
                                     pesos=self.pesos[i],
                                     biases=None)

        clasificador.agregarCapa(unidadesEntrada=self.capas[-1].n_visible,
                                 unidadesSalida=self.capas[-1].n_hidden,
                                 clasificacion=True,
                                 activacion=activaciones[-1],
                                 pesos=self.pesos[-1],
                                 biases=None)

        T = temporizador()
        inicio = T.tic()

        costoTRN, costoVAL, costoTST, costoTST_final = 0.0,0.0,0.0,0.0
        costoTRN, costoVAL, costoTST, costoTST_final = clasificador.entrenar(trainSet=datos[0],
                                                             validSet=datos[1],
                                                             testSet=datos[2],
                                                             batch_size=tambatch)

        final = T.toc()
        print("Tiempo total para el ajuste de DBN: {}".format(T.transcurrido(inicio, final)))

        # se guardan los estadisticos
        self._guardar(diccionario={'costoTRN':costoTRN, 'costoVAL':costoVAL,'costoTST':costoTST})

        # dibujar estadisticos
        clasificador.dibujarEstadisticos(mostrar=False, guardar=clasificador.ruta+'estadisticosMLP')
        clasificador.score(datos=datos[2], guardar='Matriz de Confusion')

        return costoTRN, costoVAL, costoTST, costoTST_final

    def guardarObjeto(self, nombreArchivo, compression='zip'):
        """
        guarda la mlp, en formato comprimido, todo el objeto
        """
        nombre = self.ruta + nombreArchivo
        save(objeto=self, filename=nombre, compression=compression)
        return 0

    def _guardar(self, nombreArchivo=None, diccionario=None, sobreescribir=False):
        """
        Almacena todos los datos en un archivo pickle que contiene un diccionario
        lo cual lo hace mas sencillo procesar luego
        """
        nombreArchivo = self.name if nombreArchivo is None else nombreArchivo
        archivo = self.ruta + nombreArchivo + '.cupydle'

        permitidas = self.datosAlmacenar._allowed_keys
        assert False not in [k in permitidas for k in diccionario.keys()], "el diccionario contiene una key no valida"

        with shelve.open(archivo, flag='w', writeback=False, protocol=2) as shelf:
            for key in diccionario.keys():
                if isinstance(self.datosAlmacenar[key],list) and not sobreescribir:
                    tmp = shelf[key]
                    tmp.append(diccionario[key])
                    shelf[key] = tmp
                    del tmp
                else:
                    shelf[key] = diccionario[key]
            shelf.close()
        return 0

    def _cargar(self, nombreArchivo=None, key=None):
        nombreArchivo = self.name if nombreArchivo is None else nombreArchivo
        archivo = self.ruta + nombreArchivo + '.cupydle'

        with shelve.open(archivo, flag='r', writeback=False, protocol=2) as shelf:
            if key is not None:
                assert key in shelf.keys(), "key no almacenada " + str(key)
                retorno = shelf[key]
            else:
                retorno = shelf.copy
            shelf.close()

        return retorno


    @staticmethod
    def load(filename, compression='zip'):
        datos = None
        datos = load_utils(filename=filename, compression=compression)
        return datos

    # redefino
    def __str__(self):
        print("Name:", self.name)
        print("Cantidad de capas:", self.n_layers)
        for i in range(0,len(self.capas)):
            print("-[" + str(i+1) + "] :")
            print(self.capas[i])
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
                print("Cargando capa: ",file) if DBN.DEBUG else None
                capas.append(RBM.load(str(file)).w.get_value())
        return capas

if __name__ == "__main__":
    assert False, str(__file__ + " No es un modulo")
