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

import numpy, os

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
from cupydle.dnn.utils import guardarHDF5, cargarHDF5, guardarSHELVE, cargarSHELVE
from cupydle.dnn.utils import temporizador


try:
    import PIL.Image as Image
except ImportError:
    import Image

# parte de las rbm
from cupydle.dnn.rbm import BRBM, GRBM
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
                lr_bvis=0.0,
                lr_bocu=0.0,
                costo_w=0.0,
                momento=0.0,
                toleranciaError=0.0,
                tipo='binaria'):
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
        self.tipo               = tipo
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
        print("Tipo de RBM:", self.tipo)
        return str("")

    @property
    def getParametrosEntrenamiento(self):

        tmpRBM = BRBM(n_visible=1, n_hidden=1).datosAlmacenar.keys()
        retorno = {}
        for key in self.__dict__.keys():
            if key in tmpRBM:
                retorno[key] = self.__dict__[key]
        del tmpRBM
        if os.path.isfile("rbmTest.cupydle"):
            os.remove("rbmTest.cupydle")
        return retorno

class DBN(object):

    DEBUG=False
    """bool: Indica si se quiere imprimir por consola mayor inforamcion.

    Util para propositos de debugging. Para acceder a el solo basta con

    Example:
        >>> from cupydle.dnn.mlp import MLP
        >>> M = MLP(...)
        >>> MLP.DEBUG = True

    """

    #DRIVER_PERSISTENCIA="shelve"
    DRIVER_PERSISTENCIA="hdf5"
    """str: Selecciona el driver para el almacenamiento.

    Las posibles elecciones son:
        * `shelve`_.
        * `h5py`_.

    Note:
        La implementacion de *shelve* requiere mas memoria RAM, aunque es mas
        sencilla y portable de entender.

    Example:
        >>> from cupydle.dnn.mlp import MLP
        >>> M = MLP(...)
        >>> MLP.DRIVER_PERSISTENCIA = "hdf5"
    """

    DBN_custom = False

    def __init__(self, numpy_rng=None, theano_rng=None, n_outs=None, nombre=None, ruta=''):
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

        :type nombre: string
        :param nombre: nombre of model, for save files in disk
        """

        self.semilla_numpy = None
        self.semilla_theano = None

        if not numpy_rng:
            self.semilla_numpy = 1234
        if not theano_rng:
            self.semilla_theano = 1234

        # allocate symbolic variables for the data
        self.x = theano.tensor.matrix('samples')  # los datos de entrada son del tipo tensor de orden dos
        self.y = theano.tensor.ivector('labels')  # las etiquetas son tensores de orden uno (ints)

        self.capas = []
        self.n_layers = 0

        if nombre is None:
            nombre="dbnTest"
        self.nombre = nombre

        self.numpy_rng  = numpy.random.RandomState(self.semilla_numpy)
        self.theano_rng = RandomStreams(seed=self.semilla_theano)

        self.ruta = ruta

        self.datosAlmacenar=self._initGuardar

    @property
    def _initGuardar(self, driver=DRIVER_PERSISTENCIA):
        """Inicializa los parametros del modelo.

        Se almacena en disco en una estructura determinada por el *driver* los
        datos. Inicialmente y por unica vez se deben almacenar dicha estructura
        en disco, para luego modificarla en el transcurso de la ejecucion.

        Args:
            driver (Opcional[DRIVER_PERSISTENCIA]): seleciona el motor de
                persistencia

        Return:
            dict: estructura de los datos almacenados.

        Note:
            Siempre que se desee persistir datos para utilizarlos en el
            transcurso de la ejecucion del modelo, deben agregarse aqui.

        Note:
            La estructura debe tener en cuenta lo siguiente para guardar:
                * flotantes, enteros y strings
                    * key: value
                * listas
                    * k: []
                * array variables o desconocidos en longitud al inicio, se debe post-procesar
                    * k: []

        """
        archivo = self.ruta + self.nombre + '.cupydle'

        # datos a guardar, es estatico, por lo que solo lo que se puede almacenar
        # debe setearse aca
        datos = {'tipo':                'dbn',
                 'nombre':              self.nombre,
                 'numpy_rng':           self.semilla_numpy,
                 'pesos':               [],
                 'pesos_iniciales':     [],
                 'theano_rng':          self.semilla_theano, # con set_value se actualiza
                 'activacionesOcultas': [],
                 'tasaAprendizaje':     0.0,
                 'regularizadorL1':     0.0,
                 'regularizadorL2':     0.0,
                 'momento':             0.0,
                 'epocas':              0.0,
                 'activacion':          'sigmoidea',
                 'toleranciaError':     0.0,
                 'costoTRN':            [],
                 'costoVAL':            [],
                 'costoTST':            [],
                 'tiempoMaximo':        0.0
                 }

        # se alamcena en disco
        self._guardar(diccionario=datos, driver=driver, nuevo=True)

        return datos

    def setParametros(self, parametros):
        self._guardar(diccionario=parametros)
        return 0

    def addLayer(self, n_visible=None, **kwargs):

        if n_visible is None:
            assert self.capas != [], "Debe prover de unidades de entrada para esta capa."
            n_visible = self.capas[-1].n_hidden

        # agrego una capa de rbm a la dbn, con los parametros que le paso
        self.capas.append(rbmParams(n_visible=n_visible, **kwargs))
        self.n_layers += 1

        return 0

    def entrenar(self, dataTrn, dataVal=None, pcd=True, guardarPesosIniciales=False, filtros=True):
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
        cantidadCapas = self.n_layers if DBN.DBN_custom else self.n_layers-1
        print("Entrenando con metodo \'custom\'") if DBN.DBN_custom else None
        for i in range(cantidadCapas):
            # deben diferenciarse si estamos en presencia de la primer capa o de una intermedia
            if i == 0:
                layer_input = self.x
                del dataTrn
            else:
                layer_input = self._cargar(clave='activacionesOcultas')[-1] # al entrada es la anterior, la que ya se entreno

            # una carpeta para alojar los datos de la capa intermedia
            directorioRbm = self.ruta + 'rbm_capa{}/'.format(i+1)
            os.makedirs(directorioRbm) if not os.path.exists(directorioRbm) else None

            # creo la red
            if self.capas[i].tipo == "binaria":
                capaRBM = BRBM(n_visible=self.capas[i].n_visible, n_hidden=self.capas[i].n_hidden, w=self.capas[i].w, ruta=directorioRbm)
            elif self.capas[i].tipo == 'gaussiana':
                capaRBM = GRBM(n_visible=self.capas[i].n_visible, n_hidden=self.capas[i].n_hidden, w=self.capas[i].w, ruta=directorioRbm)
            else:
                raise NotImplementedError('RBM no implementada')
            # configuro la capa, la rbm
            # si un parametro es None lo paso como una lista vacia...
            capaRBM.setParametros({k:v if v is not None else [] for k, v in self.capas[i].getParametrosEntrenamiento.items()})

            # se entrena capa por capa
            print("Entrenando la capa:", i+1)
            self._guardar(diccionario={'pesos_iniciales':capaRBM.getW}) if guardarPesosIniciales else None

            capaRBM.entrenamiento(data=layer_input,
                                  tamMiniBatch=self.capas[i].tamMiniBatch,
                                  pcd=pcd,
                                  gibbsSteps=self.capas[i].pasosGibbs,
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
        return final-inicio
    # FIN TRAIN

    def ajuste(self, datos, listaPesos=None, fnActivacion='sigmoidea', tambatch=10, semillaRandom=None):
        """
        construye un perceptron multicapa, y ajusta los pesos por medio de un
        entrenamiento supervisado.
        """
        if listaPesos is None:
            print("Cargando los pesos almacenados...") if DBN.DEBUG else None
            pesos = self._cargar(clave='pesos')
            if pesos==[]:
                print("PRECAUCION!!!, esta ajustando la red sin entrenarla!!!") if DBN.DEBUG else None
                pesos = [None] * self.n_layers
        else:
            pesos = listaPesos

        activaciones = []
        if fnActivacion is not None:
            if isinstance(fnActivacion, list):
                assert len(fnActivacion) == len(pesos), "No son suficientes funciones de activacion"
                activaciones = fnActivacion
            else:
                activaciones = [fnActivacion] * len(pesos)
        else:
            assert False, "Debe proveer una funcion de activacion"


        # se construye un mlp para el ajuste fino de los pesos con entrenamiento supervisado
        clasificador = MLP(clasificacion=True,
                           rng=semillaRandom,
                           ruta=self.ruta)

        ## solo cargo las keys del ajuste cargadas en las DBN al MLP buscando coincidencias y filtrando lo que no interesa
        tmp = clasificador.parametrosEntrenamiento
        clasificador.setParametroEntrenamiento({k: self._cargar(clave=k) for k in tmp})
        del tmp

        # cargo en el perceptron multicapa los pesos en cada capa
        # como el fit es de clasificacion, las primeras n-1 capas son del tipo
        # 'logisticas' luego la ultima es un 'softmax'
        #assert False, "aca entra para cuando es custom pero no para el caso normal de una sola capa... debe entrar siempre"
        cantidadPesos = len(pesos)-1 if DBN.DBN_custom else len(pesos)
        for i in range(0,cantidadPesos):
            clasificador.agregarCapa(unidadesEntrada=self.capas[i].n_visible,
                                     unidadesSalida=self.capas[i].n_hidden,
                                     clasificacion=False,
                                     activacion=activaciones[i],
                                     pesos=pesos[i],
                                     biases=None)

        if DBN.DBN_custom:
            # en este tipo carga la red y entrena con los pesos tal cual fue preentrenada.
            print("Ajustando con metodo \'custom\'")
            clasificador.agregarCapa(unidadesEntrada=self.capas[-1].n_visible,
                                     unidadesSalida=self.capas[-1].n_hidden,
                                     clasificacion=True,
                                     activacion=activaciones[-1],
                                     pesos=pesos[-1],
                                     biases=None)
        else:
            # w = numpy.zeros((self.capas[-1].n_visible, self.capas[-1].n_hidden),dtype=theanoFloat)
            clasificador.agregarCapa(unidadesEntrada=self.capas[-1].n_visible,
                                     unidadesSalida=self.capas[-1].n_hidden,
                                     clasificacion=True,
                                     activacion=activaciones[-1],
                                     pesos=None,
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

        return costoTRN, costoVAL, costoTST, costoTST_final, final-inicio

    def _guardar(self, diccionario, driver=DRIVER_PERSISTENCIA, nombreArchivo=None, nuevo=False):
        """Interfaz de almacenamiento para la persistencia de los datos del modelo.

        En dicha intefaz se optan por los distintos metodos de alamcenamiento
        disponibles.

        Entre ellos se encuentran:

            * `shelve`_ en la funcion :func:`cupydle.dnn.utils.guardarSHELVE`
            * `h5py`_ en la funcion :func:`cupydle.dnn.utils.guardarHDF5`

        Args:
            diccionario (dict): clave, nombre objeto; value, dato.
            driver ([Opcional[str]): seleccion del motor de persistencia.
            nombreArchivo (Opcional[str]): ruta+nombre del archivo donde persistir.
            nuevo (Opcional[bool]): Solo para el caso de inicializacion de la estructura.

        See Also:

            Almacenamiento con Shelve: :func:`cupydle.dnn.utils.guardarSHELVE`

            Almacenameinto con HDF5: :func:`cupydle.dnn.utils.guardarHDF5`

        Example:
            >>> D = DBN(...)
            >>> D._guardar({"nombre":"dbn"})
        """

        nombreArchivo = self.nombre if nombreArchivo is None else nombreArchivo
        nombreArchivo = self.ruta + nombreArchivo + '.cupydle'

        if driver == 'pickle':
            raise NotImplementedError("Funcion no implementada")
        elif driver == 'shelve':
            try:
                guardarSHELVE(nombreArchivo=nombreArchivo, valor=diccionario, nuevo=nuevo)
            except MemoryError as e:
                print("Error al guardar el modelo DBN, por falta de memoria en el Host " + str(e))
            except KeyError as e:
                print("Error sobre la clave... no es posible guardar " + str(e))
            except BaseException as e:
                print("Ocurrio un error desconocido al guardar!! no se almaceno nada " + str(e))
                raise Exception

        elif driver == 'hdf5':
            try:
                guardarHDF5(nombreArchivo=nombreArchivo, valor=diccionario, nuevo=nuevo)
            except MemoryError as e:
                print("Error al guardar el modelo DBN, por falta de memoria en el Host " + str(e))
                print(diccionario)
                raise Exception
            except KeyError as e:
                print("Error sobre la clave... no es posible guardar " + str(e))
                print(diccionario)
                raise Exception
            except BaseException as e:
                print("Ocurrio un error desconocido al guardar!! no se almaceno nada " + str(e))
                print(diccionario)
                raise Exception
        else:
            raise NotImplementedError("No se reconoce el driver de almacenamiento")

        return 0

    def _cargar(self, clave=None, driver=DRIVER_PERSISTENCIA, nombreArchivo=None):
        """Interfaz de recuperacion de los datos almacenados

        En dicha intefaz se optan por los distintos metodos de alamcenamiento
        disponibles.

        Entre ellos se encuentran:

        * `shelve`_ en la funcion :func:`cupydle.dnn.utils.guardarSHELVE`
        * `h5py`_ en la funcion :func:`cupydle.dnn.utils.guardarHDF5`

        Args:
            clave (Opcional[(str, list(str), None)]):
                clave, nombre objeto;

                * str: recupera unicamente le objeto
                * list(str): recupera una cantidad de objetos.
                * None: recuepra todos los objetos (precaucion MEMORIA RAM).
            driver ([Opcional[str]): seleccion del motor de persistencia.
            nombreArchivo (Opcional[str]): ruta+nombre del archivo donde persistir.

        Return:
            numpy.ndarray, str, dict, list: segun la estructura de como fue almacenado.

        Example:
            >>> D = DBN(...)
            >>> D._cargar(clave='nombre')
            >>> "dbn"
        """
        nombreArchivo = self.nombre if nombreArchivo is None else nombreArchivo
        nombreArchivo = self.ruta + nombreArchivo + '.cupydle'

        datos = None

        if driver == 'pickle':
            raise NotImplementedError("Funcion no implementada")
        elif driver == 'shelve':
            try:
                datos = cargarSHELVE(nombreArchivo=nombreArchivo, clave=clave)
            except MemoryError:
                print("Error al cargar el modelo DBN, por falta de memoria en el Host")
            except KeyError:
                print("Error sobre la clave... no es posible cargar")
            except BaseException as e:
                assert False, "Ocurrio un error desconocido al cargar!! " + str(e)

        elif driver == 'hdf5':
            try:
                datos = cargarHDF5(nombreArchivo=nombreArchivo, clave=clave)
            except MemoryError:
                print("Error al cargar el modelo DBN, por falta de memoria en el Host")
            except KeyError as e:
                print("Error sobre la clave... no es posible cargar " +str(e))
                raise Exception
            except BaseException as e:
                print("Ocurrio un error desconocido al cargar!! " + str(e))
                raise Exception
        else:
            raise NotImplementedError("No se reconoce el driver de almacenamiento")

        return datos

    def guardarObjeto(self, nombreArchivo, compression='zip'):
        """
        guarda la mlp, en formato comprimido, todo el objeto
        """
        nombre = self.ruta + nombreArchivo
        save(objeto=self, filename=nombre, compression=compression)
        return 0

    @staticmethod
    def load(filename, compression='zip'):
        datos = None
        datos = load_utils(filename=filename, compression=compression)
        return datos

    # redefino
    def __str__(self):
        print("Name:", self.nombre)
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
        print("Name:", self.nombre)
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
    #assert False, str(__file__ + " No es un modulo")
    pass
