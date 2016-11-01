# -*- coding: utf-8 -*-

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"

"""
Implementacion de una red multi-capa en GP-GPU/CPU (Theano).
"""
# dependencias internas
import os, sys, time, shelve, numpy
from numpy.random import RandomState as npRandom
from warnings import warn

# dependecinas de terceros
import theano

# dependencias propias
from cupydle.dnn.capas import Capa, CapaClasificacion
from cupydle.dnn.stops import criterios
from cupydle.dnn.graficos import dibujarCostos, dibujarMatrizConfusion
from cupydle.dnn.utils import guardarHDF5, cargarHDF5, guardarSHELVE, cargarSHELVE
from cupydle.dnn.utils import RestrictedDict, save, load as load_utils
from cupydle.dnn.utils_theano import shared_dataset, gpu_info, calcular_chunk

class MLP(object):
    """Clase ``Abstracta`` para la implementacion de Modelos de
    regresion/clasificacion multicapa. Tambien conocidos como
    ``Multilayer Perceptrons``.
    Para un modelo de dos capas, matematicamente se puede describir como:

    .. math::

        f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x)))

    donde :math:`x` es la entrada y el superindice indica la capa, :math:`b, W` el bias y pesos respectivamente.
    Por ultimo :math:`G, s` son las funciones de activacion para cada capa.

    Note:
        Esta clase **NO** debe ser implementada. Para mas informacion acuda a
        `aqui`_.

    .. _aqui: https://en.wikipedia.org/wiki/Multilayer_perceptron

    Args:
        clasificacion (Opcional[bool]): El modelo creado sera utilizado para
            tareas de regresion o clasificacion.

        rng (Opcional[int]): Semilla para el generador de nuemros aleatorios.
            Inicializacion de los valores para los pesos psinapticos y biases.

        ruta (Opcional[str]): Directorio de destino para el almacenamiento del
            modelo y archivo temporales.

        nombre (Opcional[str]): Nombre del modelo.

    """

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

    def __init__(self, clasificacion=True, rng=None, ruta='', nombre=None):

        # semilla para el random
        self.rng_seed = 1234 if rng is None else rng
        self.rng = (npRandom(self.rng_seed) if rng is None else npRandom(self.rng_seed))

        # la red es para clasificacion o regresion?
        self.clasificacion = clasificacion

        # se almacenan las layers por la que esta compuesta la red.
        self.capas = []

        # parametros para el entrenamiento
        self.cost   = 0.0
        self.L1     = 0.0
        self.L2_sqr = 0.0

        # se guarda la raiz del grafo para theano
        self.x = theano.tensor.matrix('x')

        # nombre del objeto, por si lo quiero diferenciar de otros en la misma
        # carpeta
        self.nombre = 'mlp' if nombre is None else nombre

        # donde se alojan los datos
        self.ruta = ruta

        self.parametrosEntrenamiento = self._initParametros
        #self._initParametros


    @property
    def _initParametros(self, driver=DRIVER_PERSISTENCIA):
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

        See Also:
            Vea tambien [Xavier10]_

        """
        nombreArchivo = self.ruta + self.nombre + '.cupydle'

        # datos a guardar, es estatico, por lo que solo lo que se puede almacenar
        # debe setearse aca
        datos = {'tipo':            'mlp',
                 'nombre':          self.nombre,
                 'numpy_rng':       self.rng_seed,
                 'activacion':      'sigmoidea',
                 'pesos':           [],
                 'pesos_iniciales': [],
                 'bias':            [],
                 'bias_iniciales':  [],
                 'tasaAprendizaje': 0.0,
                 'regularizadorL1': 0.0,
                 'regularizadorL2': 0.0,
                 'momento':         0.0,
                 'epocas':          0.0,
                 'toleranciaError': 0.0,
                 'tiempoMaximo':    0.0,
                 'costoTRN':        [],
                 'costoVAL':        [],
                 'costoTST':        []
                 }

        # se alamcena en disco
        self._guardar(diccionario=datos, driver=driver, nuevo=True)

        # necesario para ejecutar entrenamiento de la dbn... se almacena un lista
        # con los "keys" para el entrenamiento..
        # excluir lo que no es entrenamiento
        excluir = ['tipo', 'nombre', 'numpy_rng', 'theano_rng', 'pesos_iniciales', 'pesos', 'bias', 'bias_iniciales', 'costoTRN', 'costoVAL', 'costoTST']
        datos = [x for x in datos.keys() if x not in excluir]
        return datos

    def setParametroEntrenamiento(self, parametros):
        """Interfaz para la fijacion de los parametros del modelo.

        Args:
            parametros (dict): diccionario con los parametros del modelo.
                Dichos parametros pueden encontrarse en :func:`_initParametros`

        Ejemplos de los posibles parametros son:
            * tasaAprendizaje
            * regularizadorL1
            * regularizadorL2
            * momento
            * epocas
            * toleranciaError

        Example:
            >>> parametros = {'tasaAprendizaje':  0.01, 'regularizadorL1':  0.001, 'regularizadorL2':  0.0, 'momento': 0.0, 'epocas': 10, 'toleranciaError':  0.08}
            >>> from cupydle.dnn.mlp import MLP
            >>> M = MLP(...)
            >>> M.setParametrosEntrenamiento(parametros)
        """
        self._guardar(diccionario=parametros)
        return 0

    def agregarCapa(self, unidadesSalida, clasificacion, unidadesEntrada=None,
                    activacion='sigmoidea', pesos=None, biases=None):
        """
        :type unidadesEntrada: int
        :param unidadesEntrada: cantidad de nueronas en la entrada, por defecto
                                es la cantidad de salida de la capa anterior

        :type unidadesSalida: int
        :param unidadesSalida: cantidad de neuronas en la salida

        :type activacion: Class activations
        :param activacion: funcion de activacion para la capa, Sigmoid, relu..

        :type pesos: numpy.ndarray
        :param pesos: matriz de pesos W a inicializar la capa

        :type biases: numpy.ndarray
        :param biases: matriz de biases b a inicializar la capa

        """
        # primer capa, es la entrada de mlp, osea x, para iniciar el arbol
        if not self.capas:
            entrada = self.x
        else:
            # la entrada es la salida de la ultima capa hasta ahora...
            entrada = self.capas[-1].activate()

        # las entradas deben ser las salidas de la anterior
        if unidadesEntrada is None:
            assert self.capas != [], "Unidades de entrada para esta capa?."
            unidadesEntrada =   self.capas[-1].getW.shape[1]
        else:
            if self.capas != []:    # solo si ya hay carga una capa
                assert unidadesEntrada ==   self.capas[-1].getW.shape[1]

        # si la capa no es de clasificacion, es de regresion logisctica
        if not clasificacion:
            capa = Capa(unidadesEntrada = unidadesEntrada,
                        unidadesSalida = unidadesSalida,
                        entrada = entrada,
                        rng = self.rng,
                        funcionActivacion = activacion,
                        W = pesos,
                        b = biases)
        else:
            capa = CapaClasificacion(unidadesEntrada = unidadesEntrada,
                                     unidadesSalida = unidadesSalida,
                                     entrada = entrada,
                                     W = pesos,
                                     b = biases)
        # se guardan las capas
        self.capas.append(capa)

        # se guardan los parametros de las capas (w,b) para el entrenamiento
        #self.params += capa.params
        # se borra forzadamente el objeto para liberar espacio
        del capa, entrada

    def costos(self, y):
        """
        :param y: etiqueta de salida

        """
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically

        assert self.clasificacion, "Funcion solo valida para tareas de clasificacion"

        # costo, puede ser el MSE o bien el logaritmo negativo de la entropia..
        costo0 = self.capas[-1].negative_log_likelihood(y)
        #costo0 = self.capas[-1].errors(y)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        costo1 = 0.0
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        costo2 = 0.0
        for capa in self.capas:
            costo1 += abs(capa.W.sum())
            costo2 += (capa.W ** 2).sum()

        self.cost   = costo0
        self.L1     = costo1
        self.L2_sqr = costo2

    def netErrors(self, y):
        return self.capas[-1].errors(y)

    def predict(self):
        assert self.clasificacion,"Funcion valida para tareas de clasificacion"
        return self.capas[-1].predict()


    def entrenar(self, trainSet, validSet, testSet, batch_size, tamMacroBatch=None, rapido=False):
        #print(self._cargar(key=None))
        #print(self._cargar(key='regularizadorL1'))
        #assert False
        assert len(self.capas) != 0, "No hay capas, <<agregarCapa()>>"

        tamMiniBatch = batch_size

        y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of
                                        # [int] labels

        trainX, trainY  = shared_dataset(trainSet)
        validX, validY  = shared_dataset(validSet)
        testX, testY    = shared_dataset(testSet)
        del trainSet, validSet, testSet

        n_train_batches = trainX.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = validX.get_value(borrow=True).shape[0] // batch_size
        n_test_batches  = testX.get_value(borrow=True).shape[0] // batch_size

        # si utilizo la gpu es necesario esto
        if theano.config.device == 'gpu':
            #memoria_dataset, memoria_por_ejemplo, memoria_por_minibatch = calcular_memoria_requerida(cantidad_ejemplos=data.shape[0], cantidad_valores=data.shape[1], tamMiniBatch=tamMiniBatch)
            #print("Mem. Disp.:", gpu_info('Mb')[0], "Mem. Dataset:", memoria_dataset, "Mem. x ej.:", memoria_por_ejemplo, "Mem. x Minibatch:", memoria_por_minibatch)

            memoria_disponible = gpu_info('Mb')[0] * 0.8
            memoria_dataset = 4 * n_train_batches*batch_size/1024./1024.
            memoria_por_ejemplo = 4 * trainX.get_value(borrow=True).shape[1]/1024./1024.
            memoria_por_minibatch = memoria_por_ejemplo * tamMiniBatch

            info = "Mem. disp.: {:> 8.5f} Mem. dataset: {:> 8.5f} Mem. x ej.: {:> 8.5f} Mem. x Minibatch: {:> 8.5f}"
            print(info.format(memoria_disponible,memoria_dataset,memoria_por_ejemplo, memoria_por_minibatch)) if MLP.DEBUG else None

            if tamMacroBatch is None:
                tamMacroBatch = calcular_chunk(memoriaDatos=memoria_dataset, tamMiniBatch=tamMiniBatch, cantidadEjemplos=n_train_batches*batch_size)

            del memoria_disponible, memoria_dataset, memoria_por_ejemplo, memoria_por_minibatch

        if theano.config.device == 'cpu':
            if tamMacroBatch is None:
                tamMacroBatch = trainX.get_value(borrow=True).shape[0]


        # necesito actualizar los costos, si no hago este paso no tengo
        # los valores requeridos
        self.costos(y)

        # actualizaciones
        updates = []
        updates = self.construirActualizaciones(costo=self.cost,
                                                actualizaciones=updates)
        costo = (self.cost +
                self._cargar(key='regularizadorL1') * self.L1 +
                self._cargar(key='regularizadorL2') * self.L2_sqr)

        train_model, validate_model, test_model = self.construirFunciones(
                                        datosEntrenamiento=(trainX, trainY),
                                        datosValidacion=(validX, validY),
                                        datosTesteo=(testX, testY),
                                        cost=costo,
                                        batch_size=batch_size,
                                        updates=updates,
                                        y=y)

        del updates, costo

        predictor = theano.function(
                        inputs=[],
                        outputs=[self.predict(),testY],
                        givens={
                            self.x: testX},
                        name='predictor'
        )
        del trainX, trainY, validX, validY, testX, testY

        unidades = [self.capas[0].getW.shape[0]]
        unidades.extend([c.getB.shape[0] for c in self.capas])
        print("Entrenando un MLP, con [{}] unidades de entrada y {} unidades por capa".format(unidades[0], str(unidades[1:])))
        print("Cantidad de ejemplos para el entrenamiento supervisado: ", n_train_batches*batch_size)
        print("Tamanio del miniBatch: ", tamMiniBatch, "Tamanio MacroBatch: ", tamMacroBatch)
        print("MEMORIA antes de iterar: ", gpu_info('Mb'), '\nEntrenando...') if MLP.DEBUG else None
        del tamMiniBatch
        epoca = 0
        mejorEpoca = 0

        epocasAiterar = self._cargar(key='epocas')
        iteracionesMax = criterios['iteracionesMaximas'](maxIter=epocasAiterar)
        toleranciaErr = criterios['toleranciaError'](self._cargar(key='toleranciaError'))

        # se almacenan por epoca los valores de los costo
        costoVALhist = numpy.inf
        costoTSThist = numpy.inf
        costoTRN = numpy.full((epocasAiterar,), numpy.Inf)
        costoVAL = numpy.full((epocasAiterar,), numpy.Inf)
        costoTST = numpy.full((epocasAiterar,), numpy.Inf)

        # inicio del entrenamiento por epocas
        continuar = True
        try:
            while continuar:
                indice = epoca
                epoca  = epoca + 1

                # entreno con todo el conjunto de minibatches
                costoTRN[indice] = numpy.mean([train_model(i) for i in range(n_train_batches)])

                # TODO solo si proveeo datos de validacion
                costoVAL[indice] = numpy.mean([validate_model(i) for i in range(n_valid_batches)])

                print(str('Epoca {:>3d} de {:>3d}, '
                        + 'Costo entrenamiento {:> 8.5f}, '
                        + 'Error validacion {:> 8.5f}').format(
                        epoca, self._cargar(key='epocas'), costoTRN[indice]*100.0, costoVAL[indice]*100.0))

                # se obtuvo un error bajo en el conjunto validacion, se prueba en
                # el conjunto de test como funciona
                if costoVAL[indice] < costoVALhist and rapido:
                    costoVALhist = costoVAL[indice]
                    mejorEpoca = epoca

                    costoTST[indice] = numpy.mean([test_model(i) for i in range(n_test_batches)])
                    costoTSThist = costoTST[indice]
                    print('|---->>Epoca {:>3d}, error test del modelo {:> 8.5f}'.format(epoca, costoTST[indice]*100.0))
                elif not rapido: # modo normal, prueba todas las veces en el conjutno test
                    costoTST[indice] = numpy.mean([test_model(i) for i in range(n_test_batches)])

                continuar = not(iteracionesMax(resultados=epoca) or toleranciaErr(costoVAL[indice]))
            # termino el while
        except (KeyboardInterrupt, SystemExit):
            print("Se finalizo el proceso de entrenamiento. Guardando los datos\n")
            try:
                time.sleep(5)
            except (KeyboardInterrupt, SystemExit):
                sys.exit(1)

        print('Entrenamiento Finalizado. Mejor puntaje de validacion: {:> 8.5f} con un performance en el test de {:> 8.5f}'.format
              (costoVAL[numpy.argmin(costoVAL)] * 100., costoTST[numpy.argmin(costoVAL)] * 100.))

        del unidades, epocasAiterar, iteracionesMax, toleranciaErr

        costoTST[indice] = numpy.mean([test_model(i) for i in range(n_test_batches)])
        costoTST_final   = costoTST[indice]
        del n_train_batches, n_valid_batches, n_test_batches
        del train_model, validate_model, test_model

        # se guardan los pesos y bias ya entrenados
        # debido a que son grandes las matrices, las separo por le consumo de memoria
        for x in self.capas:
            self._guardar(diccionario={'pesos':x.getW})
            self._guardar(diccionario={'bias':x.getB})

        # se guardan los estadisticos
        self._guardar(diccionario={'costoTRN':costoTRN, 'costoVAL':costoVAL,'costoTST':costoTST})

        #print("reales", predictor()[1][0:25])
        #print("predic", predictor()[0][0:25])

        return costoTRN, costoVAL, costoTST, costoTST_final

    def dibujarEstadisticos(self, **kwargs):
        """Realiza los plots sobre los costos del modelo.

        Args:
            **kwargs: Argumentos que se pasan directamente a la funcion
                :func:`~cupydle.dnn.graficos.dibujarCostos`

        Note:
            Los argumentos son para el formato del dibujo.

        Example:
            >>> M = MLP(...)
            >>> M.dibujarEstadisticos(mostrar=False, guardar='estadisticosMLP')
        """

        # hay diferencias segun como se almacenan los datos, ya que los arrays
        #de costos son variables (no se sabe el tamanio al crear el MLP, sino
        #al entrenarlo a lo sumo) es por ello que se almacena dentro de un grupo
        # para hdf5
        costoTRN = self._cargar(key='costoTRN') if MLP.DRIVER_PERSISTENCIA == "shelve" else self._cargar(key='costoTRN')[0]
        costoVAL = self._cargar(key='costoVAL') if MLP.DRIVER_PERSISTENCIA == "shelve" else self._cargar(key='costoVAL')[0]
        costoTST = self._cargar(key='costoTST') if MLP.DRIVER_PERSISTENCIA == "shelve" else self._cargar(key='costoTST')[0]
        dibujarCostos(costoTRN=costoTRN, costoVAL=costoVAL, costoTST=costoTST, **kwargs)
        return 0

    def construirActualizaciones(self, costo, actualizaciones):

        cost = (costo +
                self._cargar(key='regularizadorL1') * self.L1 +
                self._cargar(key='regularizadorL2') * self.L2_sqr)

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        #self.params = parametros [W,b] "shared" de cada capa de la red
        parametrosXcapa = []
        for c in self.capas:
            parametrosXcapa.extend([c.W, c.b])
        #gparams = [theano.tensor.grad(cost, param) for param in self.params]
        gparams = [theano.tensor.grad(cost, param) for param in parametrosXcapa]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        #updates = [ (param, param - self._cargar(key='tasaAprendizaje') * gparam) for param, gparam in zip(self.params, gparams) ]
        updates = [ (param, param - self._cargar(key='tasaAprendizaje') * gparam) for param, gparam in zip(parametrosXcapa, gparams) ]

        # si ya vienen con actualizaciones (updates previas)
        #actualizaciones.append(updates)
        #actualizaciones += updates
        actualizaciones.extend(updates)
        return actualizaciones

    def construirFunciones(self, datosEntrenamiento, datosValidacion,
                           datosTesteo, cost, batch_size, updates, y):
        """
        por cuestiones de legibilidad lo pase a una funcion aparte la construccion
        de las funciones
        """

        trainX, trainY = datosEntrenamiento
        validX, validY = datosValidacion
        testX, testY = datosTesteo


        # allocate symbolic variables for the data
        index = theano.tensor.lscalar() # index to a [mini]batch

        test_model = theano.function(
                                    inputs=[index],
                                    outputs=self.netErrors(y),
                                    givens={
                                            self.x: testX[index * batch_size:(index + 1) * batch_size],
                                            y: testY[index * batch_size:(index + 1) * batch_size]
                                            },
                                    name='test_model'
        )

        validate_model = theano.function(
                                        inputs  = [index],
                                        outputs = self.netErrors(y),
                                        givens = {
                                            self.x: validX[index * batch_size:(index + 1) * batch_size],
                                            y: validY[index * batch_size:(index + 1) * batch_size]
                                        },
                                        name = 'validate_model'
        )

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
                                    inputs = [index],
                                    outputs = cost,
                                    updates = updates,
                                    givens = {
                                        self.x: trainX[index * batch_size: (index + 1) * batch_size],
                                        y: trainY[index * batch_size: (index + 1) * batch_size]
                                    },
                                    name = 'train_model'
        )
        return train_model, validate_model, test_model


    def score(self, datos, normalizar=False, mostrar=False, guardar=None):
        """
        matriz de confusion
        """
        testX, testY = datos
        cantidad_clases = len(numpy.bincount(testY))
        matriz = numpy.zeros((cantidad_clases, cantidad_clases), dtype=numpy.int32)

        score_model = theano.function(inputs=[],
                                    outputs=self.predict(),
                                    givens={self.x: testX},
                                    name='score_model')

        salida = score_model()

        for idx in range(len(salida)):
            #matriz[salida[idx], testY[idx]] +=1

            # fila=real; columna=prediccion
            matriz[testY[idx],salida[idx]] +=1

        titulo = 'Matriz de Confusion Normalizada' if normalizar else 'Matriz de Confusion'

        class_names = list(range(cantidad_clases))
        dibujarMatrizConfusion(matriz, clases=class_names, titulo='Matriz de Confusion', axe=None, mostrar=True, guardar=self.ruta + guardar)

        return matriz

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
            >>> M = MLP(...)
            >>> M._guardar({"nombre":"mlp"})
        """

        nombreArchivo = self.nombre if nombreArchivo is None else nombreArchivo
        nombreArchivo = self.ruta + nombreArchivo + '.cupydle'

        if driver == 'pickle':
            raise NotImplementedError("Funcion no implementada")
        elif driver == 'shelve':
            try:
                guardarSHELVE(nombreArchivo=nombreArchivo, valor=diccionario, nuevo=nuevo)
            except MemoryError as e:
                print("Error al guardar el modelo MLP, por falta de memoria en el Host " + str(e))
            except KeyError as e:
                print("Error sobre la clave... no es posible guardar " + str(e))
            except BaseException as e:
                print("Ocurrio un error desconocido al guardar!! no se almaceno nada " + str(e))

        elif driver == 'hdf5':
            try:
                guardarHDF5(nombreArchivo=nombreArchivo, valor=diccionario, nuevo=nuevo)
            except MemoryError as e:
                print("Error al guardar el modelo MLP, por falta de memoria en el Host " + str(e))
                print(diccionario)
            except KeyError as e:
                print("Error sobre la clave... no es posible guardar " + str(e))
                print(diccionario)
            except BaseException as e:
                print("Ocurrio un error desconocido al guardar!! no se almaceno nada " + str(e))
                print(diccionario)
        else:
            raise NotImplementedError("No se reconoce el driver de almacenamiento")

        return 0

    def _cargar(self, key=None, driver=DRIVER_PERSISTENCIA, nombreArchivo=None):
        """Interfaz de recuperacion de los datos almacenados

        En dicha intefaz se optan por los distintos metodos de alamcenamiento
        disponibles.

        Entre ellos se encuentran:

        * `shelve`_ en la funcion :func:`cupydle.dnn.utils.guardarSHELVE`
        * `h5py`_ en la funcion :func:`cupydle.dnn.utils.guardarHDF5`

        Args:
            key (Opcional[(str, list(str), None)]):
                clave, nombre objeto;

                * str: recupera unicamente le objeto
                * list(str): recupera una cantidad de objetos.
                * None: recuepra todos los objetos (precaucion MEMORIA RAM).
            driver ([Opcional[str]): seleccion del motor de persistencia.
            nombreArchivo (Opcional[str]): ruta+nombre del archivo donde persistir.

        Return:
            numpy.ndarray, str, dict, list: segun la estructura de como fue almacenado.

        Example:
            >>> M = MLP(...)
            >>> M._cargar(key='nombre')
            >>> "mlp"
        """
        nombreArchivo = self.nombre if nombreArchivo is None else nombreArchivo
        nombreArchivo = self.ruta + nombreArchivo + '.cupydle'

        datos = None

        if driver == 'pickle':
            raise NotImplementedError("Funcion no implementada")
        elif driver == 'shelve':
            try:
                datos = cargarSHELVE(nombreArchivo=nombreArchivo, clave=key)
            except MemoryError:
                print("Error al cargar el modelo MLP, por falta de memoria en el Host")
            except KeyError:
                print("Error sobre la clave... no es posible cargar")
            except BaseException as e:
                assert False, "Ocurrio un error desconocido al cargar!! " + str(e)

        elif driver == 'hdf5':
            try:
                datos = cargarHDF5(nombreArchivo=nombreArchivo, clave=key)
            except MemoryError:
                print("Error al cargar el modelo MLP, por falta de memoria en el Host")
            except KeyError as e:
                print("Error sobre la clave... no es posible cargar " +str(e))
            except BaseException as e:
                assert False, "Ocurrio un error desconocido al cargar!! " + str(e)
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
