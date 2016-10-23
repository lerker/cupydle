# -*- coding: utf-8 -*-

"""Implementacion de Maquina de Boltzmann Restringidas en CPU/GP-GPU"""

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"

# dependencias internas
import sys, numpy, shelve
from abc import ABC, abstractmethod
from warnings import warn

# dependencias de terceros
import theano
#from theano.tensor.shared_randomstreams import RandomStreams  #random seed CPU
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams # GPU
## ultra rapido
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# dependecias propias
from cupydle.dnn.unidades import UnidadBinaria, UnidadGaussiana
from cupydle.dnn.loss import errorCuadraticoMedio
from cupydle.dnn.utils_theano import gpu_info, calcular_chunk, calcular_memoria_requerida
from cupydle.dnn.utils import temporizador, RestrictedDict, save
from cupydle.dnn.graficos import imagenTiles, dibujarCostos


theanoFloat  = theano.config.floatX

try:
    import PIL.Image as Image
except ImportError:
    import Image

class RBM(ABC):
    """Clase ``Abstracta`` para la implementacion de las Maquinas de Boltzmann
    Restringidas.

    Note:
        Esta clase `NO` debe ser implementada.

    Parameters:
        n_visible (int): Cantidad de neuronas visibles.

        n_hidden (int): Cantidad de neuronas ocultas.

        w (Opcional[numpy.ndarray]): Matriz de pesos, conexiones psinapticas entre
            neuronas visibles y ocultas. Forma `(n_visivles, n_hidden)`

        w_init (Opcional[str]): Posibles: *uniforme*, *custom*. Default `uniforme`
            Eleccion del tipo de inicializacion para la matriz de pesos `W`.

            `uniforme`: metodo de inicializacion uniforme.

            .. math:: W = [\dfrac{-1}{n\_visible}, \dfrac{1}{n\_visible}]


            `custom`: metodo de inicializacion mejorada.

            .. math:: \pm 4 \cdot \sqrt{\dfrac{6}{n\_visible + n\_hidden}}

        visbiases (Opcional[shared-theano, numpy.ndarray]): Seteo de los biases visibles.
            Forma (n_visible,).

        hidbiases (Opcional[shared-theano, numpy.ndarray]): Seteo de los biases ocultos.
            Forma (n_hidden,).

        numpy_rnd (Opcional[int]): Semilla para la generacion de numeros aleatorios
            `numpy`.

        theano_rnd (opcional[int]): Semilla para la generacion de numeros aleatorios
            theano`.

        nombre (Opcional[str]): Nombre de directorio.

        ruta (Opcional[str]): Directorio absoluto para el almacenamiento del modelo
            y ejecucion.

    """

    DEBUG=False
    """bool: Modo de mas detalles.

    Example:
        >>> from cupydle.dnn.rbm import RBM
        >>> R = RBM(...)
        >>> RBM.DEBUG = True
    """

    optimizacion_hinton=False
    """bool: Modo estandar de entrenamiento para las RBM.

    Example:
        >>> from cupydle.dnn.rbm import RBM
        >>> R = RBM(...)
        >>> RBM.optimizacion_hinton = True
    """

    def __init__(self,
                 n_visible,
                 n_hidden,
                 w=None,
                 w_init='uniforme',
                 visbiases=None,
                 hidbiases=None,
                 numpy_rng=None,
                 theano_rng=None,
                 nombre=None,
                 ruta=''):

        self.n_visible = n_visible
        self.n_hidden  = n_hidden

        # create a number generator (fixed) for test NUMPY
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        # create a number generator (fixed) for test THEANO
        if theano_rng is None:
            #theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            theano_rng = RandomStreams(seed=1234)

        if w is None:
            w = self._initW(numpy_rng=numpy_rng, metodo=w_init)

        if visbiases is None:
            visbiases = self._initBiasVisible

        if hidbiases is None:
            hidbiases = self._initBiasOculto

        # w es una matriz de numpy pasada a la funcion?
        if w is not None and isinstance(w, numpy.ndarray):
            w = theano.shared(value=w, name='w', borrow=True)

        ########
        self.visbiases  = visbiases
        self.hidbiases  = hidbiases
        self.w          = w
        self.numpy_rng  = numpy_rng
        self.theano_rng = theano_rng

        # funcion de activacion sigmodea para cada neurona (porbabilidad->binomial)
        self.unidadesVisibles = None
        self.unidadesOcultas  = None

        # buffers para el almacenamiento temporal de las variables
        # momento-> incrementos, historicos
        self.vishidinc   = theano.shared(value=numpy.zeros(shape=(self.n_visible, self.n_hidden), dtype=theanoFloat), name='vishidinc')
        self.hidbiasinc  = theano.shared(value=numpy.zeros(shape=(self.n_hidden), dtype=theanoFloat), name='hidbiasinc')
        self.visbiasinc  = theano.shared(value=numpy.zeros(shape=(self.n_visible), dtype=theanoFloat), name='visbiasinc')

        # seguimiento del grafo theano, root
        self.x = theano.tensor.matrix(name="x")

        self.nombre = 'rbmTest' if nombre is None else nombre

        self.ruta = ruta

        # los parametros a guardar del todo el modelo
        self.datosAlmacenar = self._initGuardar
    # END INIT

    @abstractmethod
    def __str__(self):
        return str("RBM abstracta")

    def _initW(self, numpy_rng, metodo='uniforme'):
        """
        Inicializacion de la matriz de pesos

        Dos metedos disponibles:
        1- Asignacion uniforme
        2- metodo bengio

        :type numpy_rng: int
        :param numpy_rng: semilla para la generacion random
        :type metodo: str
        :param metodo: seleccion del metodo de inicializacion (uniform, custom)
        """
        if metodo=='uniforme':
            piso = -1.* (1. / self.n_visible)
            techo = piso * (-1.)
            _w = numpy.asarray(
                numpy.random.uniform(
                    low=piso, high=techo, size=(self.n_visible, self.n_hidden)), dtype=theanoFloat)
        elif metodo=='custom':
            _w = numpy.asarray(
                numpy_rng.uniform(
                    low= -4 * numpy.sqrt(6. / (self.n_visible + self.n_hidden)),
                    high= 4 * numpy.sqrt(6. / (self.n_visible + self.n_hidden)),
                    size=(self.n_visible, self.n_hidden)),
                dtype=theanoFloat)
        else:
            raise Exception("Metodo de inicializacion de los pesos desconocido: "
                     + str(metodo))

        # se pasa el buffer al namespace de la GPU
        w = theano.shared(value=_w, name='w', borrow=True)
        # se libera memoria de la CPU
        del _w
        return w

    @property
    def _initBiasVisible(self):
        _visbiases = numpy.zeros(shape=(self.n_visible), dtype=theanoFloat)
        visbiases = theano.shared(value=_visbiases, name='visbiases', borrow=True)
        del _visbiases
        return visbiases

    @property
    def _initBiasOculto(self):
        _hidbiases = numpy.zeros(shape=(self.n_hidden), dtype=theanoFloat)
        hidbiases = theano.shared(value=_hidbiases, name='hidbiases', borrow=True)
        del _hidbiases
        return hidbiases

    @property
    def _initGuardar(self):
        """
        inicializa los campos para la persistencia de los datos en disco,
        por ahora se restringen los parametros en un diccionario, luego se
        guarda segun el driver utilizado
        """
        archivo = self.ruta + self.nombre + '.cupydle'

        # datos a guardar, es estatico, por lo que solo lo que se puede almacenar
        # debe setearse aca
        datos = {'tipo':                self.__str__(),
                 'nombre':              self.nombre,
                 'numpy_rng':           self.numpy_rng,
                 'theano_rng':          self.theano_rng.rstate, # con set_value se actualiza
                 'w':                   None,
                 'biasVisible':         None,
                 'biasOculto':          None,
                 'w_inicial':           None,
                 'biasVisible_inicial': None,
                 'biasOculto_inicial':  None,
                 'lr_pesos':            0.0,
                 'lr_bvis':             0.0,
                 'lr_bocu':             0.0,
                 'costo_w':             0.0,
                 'momento':             0.0,
                 'epocas':              0.0,
                 'dropout':             1.0,
                 'diffEnergiaTRN':      None,
                 'errorReconsTRN':      None,
                 'mseTRN':            None,
                 }
        # diccionario restringido en sus keys
        almacenar = RestrictedDict(datos)
        for k,v in datos.items():
            almacenar[k]=v

        del datos

        # 'r'   Open existing database for reading only (default)
        # 'w' Open existing database for reading and writing
        # 'c' Open database for reading and writing, creating it if it doesn’t exist
        # 'n' Always create a new, empty database, open for reading and writing
        with shelve.open(archivo, flag='n', writeback=False) as shelf:
            for key in almacenar.keys():
                shelf[key] = almacenar[key]
            shelf.close()

        del shelf, archivo

        return almacenar

    def _guardar(self, driver='shelve', nombreArchivo=None, diccionario=None):
        """
        interfaz de almacenamiento, se eligen los metodos disponibles por el
        driver

        :type driver: str
        :param driver: seleccion del driver para persistencia, pickle/shelve
        :type nombreArchivo: str
        :param nombreArchivo: nombre del archivo a persistir
        :type diccionario: dict
        :param diccionario: diccionario con los valores a alamacenar
        """

        nombreArchivo = self.nombre if nombreArchivo is None else nombreArchivo
        nombreArchivo = self.ruta + nombreArchivo + '.cupydle'

        if driver == 'pickle':
            raise NotImplementedError("Funcion no implementada")
        elif driver == 'shelve':
            try:
                self._guardarShelve(nombreArchivo=nombreArchivo, diccionario=diccionario)
            except MemoryError:
                print("Error al guardar el modelo RBM, por falta de memoria en el Host")
        else:
            raise NotImplementedError("No se reconoce el driver de almacenamiento")

        return 0

    def _guardarShelve(self, nombreArchivo, diccionario):
        """
        almacenamiento del siccionario en disco, no se repiten los keys de shelve

        :type nombreArchivo: str
        :param nombreArchivo: nombre del archivo a persistir
        :type diccionario: dict
        :param diccionario: diccionario con los valores a alamacenar
        """
        permitidas = self.datosAlmacenar._allowed_keys
        assert False not in [k in permitidas for k in diccionario.keys()], "el diccionario contiene una key no valida"

        with shelve.open(nombreArchivo, flag='w', writeback=False, protocol=2) as shelf:
            for key in diccionario.keys():
                if isinstance(self.datosAlmacenar[key],list):
                    tmp = shelf[key]
                    tmp.append(diccionario[key])
                    shelf[key] = tmp
                    del tmp
                else:
                    shelf[key] = diccionario[key]
            shelf.close()
        return 0

    def _cargar(self, driver='shelve', nombreArchivo=None, clave=None):
        """
        interfaz de carga, se eligen los metodos disponibles por el driver

        :type driver: str
        :param driver: seleccion del driver para persistencia, pickle/shelve
        :type nombreArchivo: str
        :param nombreArchivo: nombre del archivo a persistir
        :type clave: str
        :param clave: clave a buscar en el diccionario (almacenado)
        """
        retorno = None

        nombreArchivo = self.nombre if nombreArchivo is None else nombreArchivo
        nombreArchivo = self.ruta + nombreArchivo + '.cupydle'

        if driver == 'pickle':
            raise NotImplementedError("Funcion no implementada")
        elif driver == 'shelve':
            retorno = self._cargarShelve(nombreArchivo=nombreArchivo, clave=clave)
        else:
            raise NotImplementedError("No se reconoce el driver de almacenamiento")

        return retorno

    def _cargarShelve(self, nombreArchivo, clave):
        """
        carga del diccionario almacenado en disco

        :type nombreArchivo: str
        :param nombreArchivo: nombre del archivo en disco
        :type clave: str
        :param clave: clave del diccionario almacenado en disco a cargar
        """

        with shelve.open(nombreArchivo, flag='r', writeback=False, protocol=2) as shelf:
            if clave is not None:
                assert clave in shelf.keys(), "clave no almacenada " + str(clave)
                retorno = shelf[clave]
            else:
                retorno = shelf.copy
            shelf.close()

        return retorno

    def setParametros(self, diccionario=None):
        """
        interfaz simple de almacenamiento para los parametros del modelo
        """
        self._guardar(diccionario=diccionario)
        return 0

    def set_w(self, w):
        """
        setter para los pesos

        type w: numpy.ndarray
        :param w: pesos, dimension (n_visibles,n_hidden)
        """
        self.w.set_value(w)
        return 0

    def set_biasVisible(self, bias):
        """
        setter para los bias visbles

        type bias: numpy.ndarray
        :param bias: biases, dimension (n_visibles,)
        """
        self.visbiases.set_value(bias)
        return 0

    def set_biasOculto(self, bias):
        """
        setter para los bias ocultos

        type bias: numpy.ndarray
        :param bias: biases, dimension (n_hidden,)
        """
        self.hidbiases.set_value(bias)
        return 0

    @property
    def getW(self):
        """
        getter para los pesos
        """
        return self.w.get_value(borrow=True)

    @property
    def getVisible(self):
        """
        getter para los bias visibles
        """
        return self.visbiases.get_value(borrow=True)

    @property
    def getOculto(self):
        """
        getter para los bias ocultos
        """
        return self.hidbiases.get_value(borrow=True)

    @property
    def printParams(self):
        """
        imprime por consola los parametros de la red
        """
        warn("No debe utilizarse printParams")
        for key, value in self.params.items():
            print('{: >20}: {: <10}'.format(str(key), str(value)))
        return 1

    def guardarModelo(self, nombreArchivo, compression='zip'):
        """
        Guarda el modelo entero (clases, metodos, etc) en formato comprimido
        para debug rapido

        :type nombreArchivo: str
        :param nombreArchivo: nombre del archivo a guardar (con ruta)
        :type compression: str
        :param compression: tipo de archivo final, zip/gzip..
        """
        nombre = self.ruta + nombreArchivo
        save(objeto=self, filename=nombre, compression=compression)
        return 0

    @staticmethod
    def load(nombreArchivo=None, method='simple', compression=None):
        """
        Carga desde archivo un objeto RBM guardado. Se distinguen los metodos
        de guardado, pickle simple o theano.

        :type nombreArchivo: string
        :param nombreArchivo: ruta completa al archivo donde se aloja la RBM

        :type method: string
        :param method: si es 'simple' se carga el objeto con los metodos estandar
                        para pickle, si es 'theano' se carga con las funciones
                        correspondientes

        :type compression: string
        :param compression: si es None se infiere la compresion segun 'nombreArchivo'
                            valores posibles 'zip', 'pgz' 'bzp2'

        url: http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
        # TODO
        """
        warn("no utilizar load")
        objeto = None
        if method != 'theano':
            from cupydle.dnn.utils import load
            objeto = load(nombreArchivo, compression)
        else:
            from cupydle.dnn.utils_theano import load
            objeto = load(nombreArchivo, compression)
        return objeto

    def dibujarFiltros(self, nombreArchivo='filtros.png', automatico=True, formaFiltro=(10,10), binary=False, mostrar=False):
        from cupydle.dnn.graficos import dibujarFiltros
        drop = self._cargar(clave='dropout')
        if drop != 0.0: # se escala segun el drop
            W = self.getW * (1.0/drop)
        else:
            W = self.getW

        #dibujarFiltros(w=W, nombreArchivo=self.ruta+nombreArchivo, automatico=automatico, formaFiltro=formaFiltro, binary=binary, mostrar=mostrar)
        dibujarFiltros(w=W, nombreArchivo=self.ruta+nombreArchivo, automatico=automatico, formaFiltro=formaFiltro, binary=binary, mostrar=mostrar)
        return 0

    def muestrearVdadoH(self, muestra_H):
        salidaLineal_V = theano.tensor.dot(muestra_H, self.w.T) + self.visbiases
        #salidaLineal_V1*=theano.tensor.cast(mask, theanoFloat)
        muestra_V, probabilidad_V = self.unidadesVisibles.activar(salidaLineal_V)

        return [salidaLineal_V, probabilidad_V, muestra_V]

    def muestrearHdadoV(self, muestra_V):
        salidaLineal_H = theano.tensor.dot(muestra_V, self.w) + self.hidbiases
        muestra_H, probabilidad_H = self.unidadesOcultas.activar(salidaLineal_H)

        drop = self._cargar(clave='dropout')
        if drop != 0.0 and drop != 1.0:
            mask = self.theano_rng.binomial(size=self.hidbiases.shape, n=1, p=drop, dtype=theanoFloat)
            salidaLineal_H = mask*salidaLineal_H
            probabilidad_H = mask*probabilidad_H
            muestra_H = mask*muestra_H

        return [salidaLineal_H, probabilidad_H, muestra_H]

    def pasoGibbsVHV(self, muestra, steps):
        # si fuera de un solo paso, se samplea las ocultas, re recupera las visibles
        # y vuelve a samplear las ocultas (aunque no se utilece el samble ultimo, solo la probabilidad)

        # un paso de CD es Visible->Hidden->Visible
        def unPaso(muestraV, w, vbias, hbias):

            salidaLineal_H, probabilidad_H, muestra_H = self.muestrearHdadoV(muestraV)
            salidaLineal_V, probabilidad_V, muestra_V = self.muestrearVdadoH(muestra_H)

            return [salidaLineal_H, probabilidad_H, muestra_H, salidaLineal_V, probabilidad_V, muestra_V]

        ( [salidaLineal_H, probabilidad_H, muestra_H, salidaLineal_V, probabilidad_V, muestra_V],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_pasoGibbsVHV')

        return ([salidaLineal_H[-1], probabilidad_H[-1], muestra_H[-1], salidaLineal_V[-1], probabilidad_V[-1], muestra_V[-1]], updates)

    def pasoGibbsHVH(self, muestra, steps):

        def unPaso(muestraH, w, vbias, hbias):
            salidaLineal_V, probabilidad_V, muestra_V = self.muestrearVdadoH(muestraH)
            salidaLineal_H, probabilidad_H, muestra_H = self.muestrearHdadoV(muestra_V)

            return [salidaLineal_V, probabilidad_V, muestra_V, salidaLineal_H, probabilidad_H, muestra_H]

        ( [salidaLineal_V, probabilidad_V, muestra_V, salidaLineal_H, probabilidad_H, muestra_H],
          updates) = theano.scan(fn           = unPaso,
                                 outputs_info = [None, None, None, None, None, muestra],
                                 non_sequences= [self.w, self.visbiases, self.hidbiases],
                                 n_steps      = steps,
                                 strict       = True,
                                 name         = 'scan_pasoGibbsHVH')
        return ([salidaLineal_V[-1], probabilidad_V[-1], muestra_V[-1], salidaLineal_H[-1], probabilidad_H[-1], muestra_H[-1]], updates)

    def energiaLibre(self, vsample):
        """
        Calcula la energia libre F(v) = - log sum_h exp(-E(v,h)).

        Parametros
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Retornos
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        wx_b = theano.tensor.dot(vsample, self.w) + self.hidbiases
        vbias_term = theano.tensor.dot(vsample, self.visbiases)
        hidden_term = theano.tensor.sum(theano.tensor.log(1 + theano.tensor.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def pseudoLikelihoodCost(self, updates=None):
        # pseudo-likelihood is a better proxy for PCD
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = theano.tensor.round(self.x)

        # calculate free energy for the given bit configuration
        fe_xi = self.energiaLibre(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = theano.tensor.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.energiaLibre(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = theano.tensor.mean(self.n_visible \
             * theano.tensor.log(theano.tensor.nnet.sigmoid(fe_xi_flip
                                                            - fe_xi)))

        # increment bit_i_idx % number as part of updates
        #TODO esto estaba, pero arroja un error (paraque si no lo uso)?
        #updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        # TODO es necesario retornar las updates?
        # ni siquiera las utiliza en el algoritmo.. medio inutil pasarla o retornarla
        return cost

    def pseudoLikelihoodCost_bengio(self, updates=None):
        # pseudo-likelihood is a better proxy for PCD
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = theano.tensor.round(self.x)

        # calculate free energy for the given bit configuration
        fe_xi = self.energiaLibre(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = theano.tensor.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.energiaLibre(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = theano.tensor.mean(self.n_visible \
             * theano.tensor.log(theano.tensor.nnet.sigmoid(fe_xi_flip
                                                            - fe_xi)))

        # increment bit_i_idx % number as part of updates
        #TODO esto estaba, pero arroja un error (paraque si no lo uso)?
        #updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        # TODO es necesario retornar las updates?
        # ni siquiera las utiliza en el algoritmo.. medio inutil pasarla o retornarla
        return cost

    def reconstructionCost(self, linearSumN):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """
        # reconstruction cross-entropy is a better proxy for CD
        crossEntropy = theano.tensor.mean(
            theano.tensor.sum(
                self.x * theano.tensor.log(theano.tensor.nnet.sigmoid(linearSumN)) +
                (1 - self.x) * theano.tensor.log(1 - theano.tensor.nnet.sigmoid(linearSumN)),
                axis=1
            )
        )
        return crossEntropy

    def reconstructionCost_MSE(self, reconstrucciones):
        warn("No se deberia utilizar mas, ver <<loss>>")
        # mean squared error, error cuadratico medio esta mal este de aca

        mse = theano.tensor.mean(
                theano.tensor.sum(
                    theano.tensor.sqr( self.x - reconstrucciones ), axis=1
                )
        )

        return mse

    def DivergenciaContrastivaPersistente(self, miniBatchSize, sharedData):
        # con dropout
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # primer termino del gradiente de la funcion de verosimilitud
        # la esperanza sobre el conjunto del dato
        salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV(self.x)

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((miniBatchSize, self.n_hidden),
                                                     dtype=theanoFloat),
                                         borrow=True)

        # aproximo el segundo termino del gradiente de la funcion de verosimilitud
        # por medio de una cadena de Gibbs
        ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_Hk, probabilidad_Hk, muestra_Hk],
         updates) = self.pasoGibbsHVH(persistent_chain, steps)


        energia_dato = theano.tensor.mean(self.energiaLibre(self.x))
        # TODO ver si le meto la salida lineal o bien la probabilidad
        energia_modelo = theano.tensor.mean(self.energiaLibre(probabilidad_Vk))

        diffEnergia = energia_dato - energia_modelo

        # construyo las actualizaciones en los updates (variables shared)
        updates = self.construirActualizaciones(
                    updatesOriginal = updates,
                    probabilidad_H0 = probabilidad_H0,
                    muestra_H0      = muestra_H0,
                    probabilidad_Vk = probabilidad_Vk,
                    probabilidad_Hk = probabilidad_Hk,
                    muestra_Hk      = muestra_Hk,
                    muestra_Vk      = muestra_Vk,
                    miniBatchSize   = miniBatchSize)

        reconstError = self.reconstructionCost(salidaLineal_Vk)

        mse = errorCuadraticoMedio(self.x, probabilidad_Vk)

        updates.append((persistent_chain, muestra_Hk))

        train_rbm = theano.function(
                        inputs  = [miniBatchIndex, steps],
                        outputs = [diffEnergia, reconstError, mse],
                        updates = updates,
                        givens  = {
                            self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_pcd'
        )

        return train_rbm

    def DivergenciaContrastiva(self, miniBatchSize, sharedData):
        steps = theano.tensor.iscalar(name='steps')         # CD steps
        miniBatchIndex = theano.tensor.lscalar('miniBatchIndex')

        # primer termino del gradiente de la funcion de verosimilitud
        # la esperanza sobre el conjunto del dato
        salidaLineal_H0, probabilidad_H0, muestra_H0 = self.muestrearHdadoV(self.x)

        # aproximo el segundo termino del gradiente de la funcion de verosimilitud
        # por medio de una cadena de Gibbs
        ([salidaLineal_Vk, probabilidad_Vk, muestra_Vk, salidaLineal_Hk, probabilidad_Hk, muestra_Hk],
         updates) = self.pasoGibbsHVH(muestra_H0, steps)

        energia_dato = theano.tensor.mean(self.energiaLibre(self.x))
        # TODO ver si le meto la salida lineal o bien la probabilidad
        energia_modelo = theano.tensor.mean(self.energiaLibre(probabilidad_Vk))

        diffEnergia = energia_dato - energia_modelo

        # construyo las actualizaciones en los updates (variables shared)
        updates = self.construirActualizaciones(
                    updatesOriginal = updates,
                    probabilidad_H0 = probabilidad_H0,
                    muestra_H0      = muestra_H0,
                    probabilidad_Vk = probabilidad_Vk,
                    probabilidad_Hk = probabilidad_Hk,
                    muestra_Hk      = muestra_Hk,
                    muestra_Vk      = muestra_Vk,
                    miniBatchSize   = miniBatchSize)

        reconstError = self.reconstructionCost(salidaLineal_Vk)

        mse = errorCuadraticoMedio(self.x, probabilidad_Vk)

        train_rbm = theano.function(
                        inputs  = [miniBatchIndex, steps],
                        outputs = [diffEnergia, reconstError, mse],
                        updates = updates,
                        givens  = {
                                    self.x: sharedData[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
                        },
            name='train_rbm_cd'
        )

        # retorna [diffEnergia, reconstError, mse] sobre el conjunto de entrenamiento
        return train_rbm

    def construirActualizaciones(self, updatesOriginal, probabilidad_H0, muestra_H0, probabilidad_Vk, probabilidad_Hk, muestra_Hk, muestra_Vk, miniBatchSize):
        """
        calcula las actualizaciones de la red sobre sus parametros w hb y vh
        con momento
        tasa de aprendizaje para los pesos, y los biases visibles y ocultos
        """
        # arreglo los parametros en caso de que no se hayan establecidos, considero las
        # tasa de aprendizaje para los bias igual al de los pesos
        assert numpy.any(self._cargar(clave='lr_pesos') != 0.0), "La tasa de aprendizaje para los pesos no puede ser nula"

        if self._cargar(clave='lr_bvis') is None or numpy.any(self._cargar(clave='lr_bvis') == 0.0):
            self._guardar(diccionario={'lr_bvis':self._cargar(clave='lr_pesos')})

        if self._cargar(clave='lr_bocu') is None or numpy.any(self._cargar(clave='lr_bocu') == 0.0):
            self._guardar(diccionario={'lr_bocu':self._cargar(clave='lr_pesos')})

        momento  = theano.tensor.cast(self._cargar(clave='momento'),  dtype=theanoFloat)
        lr_pesos = theano.tensor.cast(self._cargar(clave='lr_pesos'), dtype=theanoFloat)
        lr_bvis  = theano.tensor.cast(self._cargar(clave='lr_bvis'),dtype=theanoFloat)
        lr_bocu  = theano.tensor.cast(self._cargar(clave='lr_bocu'),dtype=theanoFloat)
        updates = []

        # las actualizaciones de hinton, en practical... utiliza solo las probabilidades y no la muestra
        # actualizacion de W
        if RBM.optimizacion_hinton:
            positiveDifference = theano.tensor.dot(self.x.T, probabilidad_H0)
            negativeDifference = theano.tensor.dot(muestra_Vk.T, probabilidad_Hk)
        else:
            positiveDifference = theano.tensor.dot(self.x.T, muestra_H0)
            negativeDifference = theano.tensor.dot(muestra_Vk.T, muestra_Hk)

        delta = positiveDifference - negativeDifference
        wUpdate = momento * self.vishidinc
        wUpdate += lr_pesos * delta / miniBatchSize
        #wUpdate *= (1.0/(dropout))
        updates.append((self.w, self.w + wUpdate))
        updates.append((self.vishidinc, wUpdate))

        # actualizacion de los bias visibles
        if RBM.optimizacion_hinton:
            visibleBiasDiff = theano.tensor.sum(self.x - probabilidad_Vk , axis=0)
        else:
            visibleBiasDiff = theano.tensor.sum(self.x - muestra_Vk , axis=0)

        biasVisUpdate = momento * self.visbiasinc
        biasVisUpdate += lr_bvis * visibleBiasDiff / miniBatchSize
        updates.append((self.visbiases, self.visbiases + biasVisUpdate))
        updates.append((self.visbiasinc, biasVisUpdate))

        # actualizacion de los bias ocultos
        if RBM.optimizacion_hinton:
            hiddenBiasDiff = theano.tensor.sum(probabilidad_H0 - probabilidad_Hk, axis=0)
        else:
            hiddenBiasDiff = theano.tensor.sum(muestra_H0 - muestra_Hk, axis=0)

        biasHidUpdate = momento * self.hidbiasinc
        biasHidUpdate += lr_bocu * hiddenBiasDiff / miniBatchSize
        updates.append((self.hidbiases, self.hidbiases + biasHidUpdate))
        updates.append((self.hidbiasinc, biasHidUpdate))

        # se concatenan todas las actualizaciones en una sola
        updates += updatesOriginal.items()

        return updates

    @abstractmethod
    def entrenamiento(self, data, tamMiniBatch=10, tamMacroBatch=None, pcd=True, gibbsSteps=1, filtros=False, printCompacto=False):
        """Metodo principal de entrenameinto para una RBM.


        Parameters:
            data (numpy-array): Conjunto de datos de entrada para el
                entrenamiento. Los datos de entrada tiene la forma (n_ejemplos,
                n_caracteristicas).

            tamMiniBatch (Opcional[int]): Cantidad de ejemplos para el
                entrenamiento del `Divergencia Cotrastiva Batch`.

            tamMacroBatch (Opcional[int]): Cantidad de ejemplos para la
                transferencia hacia la GP-GPU.
                Si es `None` se calcula de forma automatica.

            pcd (Opcional[bool]): Seleccion de entrenamiento por `Divergencia
                Constrastiva Persistente`.

            gibbsSteps (Opcional[int]): Cantidad de pasos de Gibbs a ejecutar
                en cada cadena de CD/PCD.

            filtros (Opcional[bool]): Grafica los filtros de los pesos por cada
                unidad oculta.

            printCompacto (Opcional[bool]): Imprime por consola de forma reducida,
                mejora de lectura aunque menos detallado.


        Example:
            Entrenamiento de una RBM tipo binaria.

        Note:
            Implementar a traves de una
            instancia de una clases hija, por ejemplo: :class:`.BRBM` en su
            metodo :meth:`~BRMB.entrenamiento`.

            >>> from cupydle.dnn.rbm import BRBM
            >>> R = BRBM(...)
            >>> R.entrenamiento(...)

        """

        # si utilizo la gpu es necesario esto
        if theano.config.device == 'gpu':
            memoria_dataset, memoria_por_ejemplo, memoria_por_minibatch = calcular_memoria_requerida(cantidad_ejemplos=data.shape[0], cantidad_valores=data.shape[1], tamMiniBatch=tamMiniBatch)

            if tamMacroBatch is None:
                    tamMacroBatch = calcular_chunk(memoriaDatos=memoria_dataset, tamMiniBatch=tamMiniBatch, cantidadEjemplos=data.shape[0])

        if theano.config.device == 'cpu':
            if tamMacroBatch is None:
                tamMacroBatch = len(data)

        tamMacroBatch = int(tamMacroBatch)
        tamMiniBatch = int(tamMiniBatch)

        # comprobaciones por si las dudas
        assert data.shape[0] % tamMacroBatch == 0
        assert tamMacroBatch % tamMiniBatch == 0

        macro_batch_count = int(data.shape[0] / tamMacroBatch)
        micro_batch_count = int(tamMacroBatch / tamMiniBatch)

        # buffer para alojar los datos de entrenamiento
        sharedData = theano.shared(numpy.empty((tamMacroBatch,) + data.shape[1:], dtype=theanoFloat), borrow=True)

        trainer = None
        if pcd:
            print("Entrenando con Divergencia Contrastiva Persistente, {} pasos de Gibss.".format(gibbsSteps))
            if RBM.optimizacion_hinton:
                print("Optimizacion de Hinton")

            trainer = self.DivergenciaContrastivaPersistente(tamMiniBatch, sharedData)
        else:
            print("Entrenando con Divergencia Contrastiva, {} pasos de Gibss.".format(gibbsSteps))
            if RBM.optimizacion_hinton:
                print("Optimizacion de Hinton")
            trainer = self.DivergenciaContrastiva(tamMiniBatch, sharedData)
        print("Unidades de visibles:",self.unidadesVisibles, "Unidades Ocultas:", self.unidadesOcultas)

        if filtros:
            # plot los filtros iniciales (sin entrenamiento)
            self.dibujarFiltros(nombreArchivo='filtros_epoca_0.pdf', automatico=True)

        epocasAiterar = self._cargar(clave='epocas')

        # se almacenan por epoca los valores de los costo
        diffEnergiaTRN = numpy.full((epocasAiterar,), numpy.Inf)
        errorReconsTRN = numpy.full((epocasAiterar,), numpy.Inf)
        mseTRN         = numpy.full((epocasAiterar,), numpy.Inf)

        finLinea = '\r' if printCompacto else '\n'

        print("Entrenando una RBM, con [{}] unidades visibles y [{}] unidades ocultas".format(self.n_visible, self.n_hidden))
        print("Cantidad de ejemplos para el entrenamiento no supervisado: ", len(data))
        print("Tamanio del MiniBatch: ", tamMiniBatch, "Tamanio MacroBatch: ", tamMacroBatch)
        if theano.config.device == 'gpu':
            print("Mem. Disp.:", gpu_info('Mb')[0], "Mem. Dataset:", memoria_dataset, "Mem. x ej.:", memoria_por_ejemplo, "Mem. x Minibatch:", memoria_por_minibatch)


        for epoch in range(0, epocasAiterar):
            indice = epoch
            epoca  = epoch + 1

            salida = [0.0,0.0,0.0] # salida del trainer, ojo que si cambia la cantindad se debe cambiar aca [diffEnergia, reconstError, mse]
            for macro_batch_index in range(0,macro_batch_count):
                sharedData.set_value(data[macro_batch_index * tamMacroBatch: (macro_batch_index + 1) * tamMacroBatch], borrow=True)
                # trainer devuelve 3 valores por llamada, mean-> axis=0 promeida cada valor y retorna una lista con los promedios (len==3)
                salida += numpy.mean([trainer(batch, gibbsSteps) for batch in range(micro_batch_count)], axis=0)

            diffEnergiaTRN[indice], errorReconsTRN[indice], mseTRN[indice] = salida / macro_batch_count

            # imprimo algo de informacion sobre la terminal
            print(str('Epoca {: >4d} de {: >4d}, Costo:{:> 8.5f}, MSE:{:> 8.5f}, EnergiaLibre:{:> 8.5f}').format(
                        epoca, epocasAiterar, errorReconsTRN[indice], mseTRN[indice], diffEnergiaTRN[indice]),
                    end=finLinea)

            if filtros:
                self.dibujarFiltros(nombreArchivo='filtros_epoca_{}.pdf'.format(epoca), automatico=True)
            # END SET
        # END epoch
        print("",flush=True) # para avanzar la linea y no imprima arriba de lo anterior

        # se guardan los estadisticos
        self._guardar(diccionario={'diffEnergiaTRN':diffEnergiaTRN, 'errorReconsTRN':errorReconsTRN, 'mseTRN':mseTRN})

        # se guardan los pesos y bias entrenados
        # se escalan segun el dropout
        drop = self._cargar(clave='dropout')
        if drop != 0.0: # se escala segun el drop
            W = self.getW * (1.0/drop)
        else:
            W = self.getW
        self._guardar(diccionario={'w': W, 'biasVisible':self.visbiases.get_value(borrow=True), 'biasOculto': self.hidbiases.get_value(borrow=True)})

        nombreArchivo=self.ruta+'estadisticos_'+self.nombre
        dibujarCostos(guardar=nombreArchivo, diffEnergiaTRN=diffEnergiaTRN, errorReconsTRN=errorReconsTRN, mseTRN=mseTRN)
        return 1

    def muestra(self, muestraV, gibbsSteps=1):
        """
        realiza la reconstruccion a partir de un ejemplo, efectuando una cadena
        de markov con n pasos de gibbs sobre la misma

        puedo pasarle un solo ejemplo o una matrix con varios de ellos por fila
        """

        if muestraV.ndim == 1:
            # es un vector, debo cambiar el root 'x' antes de armar el grafo
            # para que coincida con la entrada
            viejoRoot = self.x
            self.x = theano.tensor.fvector('x')

        data  = theano.shared(numpy.asarray(a=muestraV, dtype=theanoFloat), borrow=True, name='datoMuestra')

        ([salidaLineal_Hk, probabilidad_Hk, muestra_Hk, salidaLineal_Vk, probabilidad_Vk, muestra_Vk], updates) = self.pasoGibbsVHV(data, gibbsSteps)

        reconstructor = theano.function(
                        inputs=[],
                        outputs=[probabilidad_Hk, muestra_Vk, muestra_Hk],
                        updates=updates,
                        #givens={self.x: data},
                        name='fn_muestra'
        )

        [probabilidad_H, muestra_V, muestra_H] = reconstructor()

        if muestraV.ndim == 1:
            # hago el swap
            self.x = viejoRoot

        del data
        return [probabilidad_H, muestra_V, muestra_H]


    def sampleo(self, data, labels=None, chains=20, samples=10, gibbsSteps=1000, patchesDim=(28,28), binary=False):
        """
        Realiza un 'sampleo' de los datos con los parametros 'aprendidos'
        El proposito de la funcion es ejemplificar como una serie de ejemplos
        se muestrean a traves de la red ejecuntado sucesivas cadenas de markov

        :type data: numpy.narray (sample, data)
        :param data: conjunto de datos a extraer las muestras

        :type labels: numpy.narray (sample,)
        :param labels: conjunto de etiquetas que se corresponden a los datos

        :type chains: int
        :param chains: cantidad de cadenas parallelas de Gibbs de las cuales se muestrea

        :type samples: int
        :param samples: cantidad de muestras a realizar sobre cada cadena

        :type gibbsSteps: int
        :param gibssSteps: catidad de pasos de gibbs a ejecutar por cada muestra

        :type patchesDim: tuple ints
        :param patchesDim: dimesion de los patches 'sampleo' de cada cadena,
                            (alto, ancho). Para mnist (28,28)

        :type binary: bool
        :param binary: la imagen de salida debe ser binarizada
        """
        #from cupydle.dnn.graficos import dibujarCadenaMuestras

        #dibujarCadenaMuestras(data=data, labels=labels, chains=chains, samples=samples, gibbsSteps=gibbsSteps, patchesDim=patchesDim, binary=binary)

        data  = theano.shared(numpy.asarray(a=data, dtype=theanoFloat), name='DataSample')
        n_samples = data.get_value(borrow=True).shape[0]

        # seleeciona de forma aleatoria donde comienza a extraer los ejemplos
        # devuelve un solo indice.. desde [0,..., n - chains]
        test_idx = self.numpy_rng.randint(n_samples - chains)

        # inicializacion de todas las cadenas... el estado persiste a traves
        # del muestreo
        cadenaFija = theano.shared(
                        numpy.asarray(data.get_value(borrow=True)
                                        [test_idx:test_idx + chains],
                                    dtype=theanoFloat
                        )
        )
        # tengo leyenda sobre la imagen?
        if labels is not None:
            lista = range(test_idx, test_idx + chains)
            print("labels: ", str(labels[lista]))

        ([_, _, _, _, probabilidad_V1, muestra_V1],
          updates) = self.pasoGibbsVHV(self.x, gibbsSteps)

        # cambiar el valor de la cadenaFija al de la reconstruccion de las visibles
        updates.update({cadenaFija: muestra_V1})

        # funcion princial
        muestreo = theano.function(inputs=[],
                                   outputs=[probabilidad_V1, muestra_V1],
                                   updates=updates,
                                   givens={self.x: cadenaFija},
                                   name='muestreo'
        )

        # dimensiones de los patches, para el mnist es (28,28)
        #ancho=28
        #alto=28
        alto, ancho = patchesDim
        imageResults = numpy.zeros(((alto+1) * samples + 1, (ancho+1) * chains - 1),
                            dtype='uint8')
        for idx in range(samples):
            #genero muestras y visualizo cada gibssSteps, ya que las muestras intermedias
            # estan muy correlacionadas, se visializa la probabilidad de activacion de
            # las unidades ocultas (NO la muestra binomial)
            probabilidad_V1, visiblerecons = muestreo()

            print(' ... plotting sample {}'.format(idx))
            imageResults[(alto+1) * idx:(ancho+1) * idx + ancho, :] \
                = imagenTiles(X=probabilidad_V1,
                              img_shape=(alto, ancho),
                              tile_shape=(1, chains),
                              tile_spacing=(1, 1)
                )

        # construct image
        image = Image.fromarray(imageResults)

        nombreArchivo = self.ruta + "samples_" + str(labels[lista])
        nombreArchivo = nombreArchivo.replace(" ", "_")

        # poner las etiquetas sobre la imagen
        watermark = False
        if watermark:
            from PIL import ImageDraw, ImageFont
            # get the ImageDraw item for this image
            draw = ImageDraw.Draw(image)
            fontsize = 15
            font = ImageFont.truetype("arial.ttf", fontsize)
            #fill = (255,255,255) # blanco
            fill = (255,255,0) # amarillo
            fill = 255
            draw.text((0, 0),text=str(labels[lista]),fill=fill,font=font)

        if binary:
            gray = image.convert('L')
            # Let numpy do the heavy lifting for converting pixels to pure black or white
            bw = numpy.asarray(gray).copy()
            # Pixel range is 0...255, 256/2 = 128
            bw[bw < 128] = 0    # Black
            bw[bw >= 128] = 255 # White
            # Now we put it back in Pillow/PIL land
            image2 = Image.fromarray(bw)
            image2.save(nombreArchivo + '_binary.pdf')

        image.save(nombreArchivo + '.pdf')

        return 0


class BRBM(RBM):
    """Maquina de Boltzmann Restringida Binaria

    Maquina de Boltzmann Restringida \"*RBM*\" Binaria; Unidades Visibles
    Binarias y Unidades Ocultas Binarias. Funcion de activacion Sigmoidea.
    """
    def __init__(self, **kwargs):
        # inicializar la clase padre
        super().__init__(**kwargs)


    def __str__(self):
        return str("RBM Binaria-Binaria")

    def entrenamiento(self,**kwargs):
        """Entrenamiento de para una RBM Binaria

        Parameters:
            data (numpy-array): Conjunto de datos de entrada para el
                entrenamiento. Los datos de entrada tiene la forma (n_ejemplos,
                n_caracteristicas).

            tamMiniBatch (Opcional[int]): Cantidad de ejemplos para el
                entrenamiento del `Divergencia Cotrastiva Batch`.

            tamMacroBatch (Opcional[int]): Cantidad de ejemplos para la
                transferencia hacia la GP-GPU.
                Si es `None` se calcula de forma automatica.

            pcd (Opcional[bool]): Seleccion de entrenamiento por `Divergencia
                Constrastiva Persistente`.

            gibbsSteps (Opcional[int]): Cantidad de pasos de Gibbs a ejecutar
                en cada cadena de CD/PCD.

            filtros (Opcional[bool]): Grafica los filtros de los pesos por cada
                unidad oculta.

            printCompacto (Opcional[bool]): Imprime por consola de forma reducida,
                mejora de lectura aunque menos detallado.


        Example:
            Entrenamiento de una RBM tipo binaria.

            >>> from cupydle.dnn.rbm import BRBM
            >>> R = BRBM(...)
            >>> R.entrenamiento(...)
        """

        self.unidadesVisibles = UnidadBinaria()
        self.unidadesOcultas = UnidadBinaria()
        super().entrenamiento(**kwargs)
        return 0

class GRBM(RBM):
    """
    Implementacion de una Maquina de Boltzmann Restringida Gaussiana
    Unidades Visibles  Gaussianas
    Unidades Ocultas   Binarias
    """
    def __init__(self, **kwargs):
        # inicializar la clase padre
        super().__init__(**kwargs)


    def __str__(self):
        return str("RBM Binaria-Guassiana")

    def entrenamiento(self,**kwargs):
        self.unidadesVisibles = UnidadGaussiana()
        self.unidadesOcultas = UnidadBinaria()
        super().entrenamiento(**kwargs)
        return 0

def test_rbm():

    datos = numpy.load('cupydle/data/DB_mnist/mnist_minmax.npz')

    entrenamiento        = datos['entrenamiento'].astype(numpy.float32)
    entrenamiento_clases = datos['entrenamiento_clases'].astype(numpy.int32)
    validacion           = datos['validacion'].astype(numpy.float32)
    validacion_clases    = datos['validacion_clases'].astype(numpy.int32)
    testeo               = datos['testeo'].astype(numpy.float32)
    testeo_clases        = datos['testeo_clases'].astype(numpy.int32)
    del datos # libera memoria

    datos = []
    datos.append((entrenamiento, entrenamiento_clases))
    datos.append((validacion, validacion_clases))
    datos.append((testeo, testeo_clases))
    del entrenamiento, entrenamiento_clases, validacion, validacion_clases
    del testeo, testeo_clases

    # construct RBM
    rbm = BRBM(n_visible=784, n_hidden=500, w_init='uniforme')
    print(rbm)


    parametros={'lr_pesos':  0.05,
                'lr_bvis':   0.05,
                'lr_bocu':   0.05,
                'momento':   0.1,
                'costo_w':   0.1,
                'dropout':   0.0} # probabilidad de actividad en la neurona, =1 todas, =0 ninguna

    rbm.setParametros(parametros)
    rbm.setParametros({'epocas':3})

    RBM.optimizacion_hinton=True
    rbm.entrenamiento(data=datos[0][0], tamMiniBatch=10, pcd=True, filtros=True, gibbsSteps=1)

    rbm.sampleo(data=datos[0][0], labels=datos[0][1])

    return 0


if __name__ == "__main__":
    #assert False, str(__file__ + " No es un modulo")
    test_rbm()


