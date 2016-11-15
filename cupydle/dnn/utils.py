# -*- coding: utf-8 -*-

import sys, random, numbers, timeit, numpy as np
import pickle, bz2, gzip, shelve, h5py

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
Funciones *utilitarios* para todo el software
"""

def mostrar_tamaniode(x, level=0):
    """Imprime el tamanio en memoria requerido para el objeto `x`

    Args:
        x (Object)
        level (int): cantidad de subniveles a entrar.

    Returns:
        imprime por pantalla el tamanio del objeto pasado

    Examples:
        >>> mostrar_tamaniode(int)
        4

    """
    print("\t" * level, x.__class__, sys.getsizeof(x), x)

    if hasattr(x, '__iter__'):
        if hasattr(x, 'items'):
            for xx in x.items():
                mostrar_tamaniode(xx, level + 1)
        else:
            for xx in x:
                mostrar_tamaniode(xx, level + 1)
    return 0


def split_data(data, fractions, seed=123):
    """
    Split data en sets en base a fractions
    :param data: list or np.array
    :param fractions: list [f_train, f_valid, f_test]
    :param seed: int
    :return: sets (e.g. train, valid, test)
    """
    # Verifico que fractions sea correcto
    # TODO: assert (sum(fractions) <= 1.0, Exception("Fracciones para conjuntos incorrectas!"))

    np.random.seed(seed)
    # indice random
    indices = np.random.permutation(data.shape[0])

    # cantidad en numero para cada conjunto
    trainix = int(data.shape[0] * fractions[0])
    validationix = int(data.shape[0] * fractions[1])
    testix = int(data.shape[0] * fractions[2])

    training_idx, validation_idx, test_idx = indices[:trainix-validationix+1], \
                                             indices[trainix-validationix+1:trainix+validationix], \
                                             indices[-testix:]
    training, validation, test = data[training_idx, :], data[validation_idx, :], data[test_idx, :]

    return training, validation, test


def vectorize_label(label, n_classes):
    """
    Dada una etiqueta genera un vector de dimension (n_classes, 1) de 0 y un 1 en la posicion de la label
    e.g: si label es '5', n_classes es '8':
    return:
    v[0]=0
    v[1]=0
    v[2]=0
    v[3]=0
    v[4]=1
    v[5]=0
    v[6]=0
    v[7]=0


    :param label:
    :param n_classes:
    :return:
    """
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)

def downCast(objeto):

    if isinstance(objeto, np.ndarray):
        if objeto.dtype == 'float64':
            objeto = objeto.astype(np.float32)
        elif objeto.dtype =='int64':
            objeto = objeto.astype(np.int32)
    elif isinstance(objeto, float):
        objeto = np.float32(objeto)
    elif isinstance(objeto, int):
        objeto = np.int32(objeto)

    return objeto

def guardarHDF5(nombreArchivo, valor, nuevo=False):
    """Metodo de almacenamiento por medio de HDF5 (h5py)

    Es mas eficiente con respecto a pickle en cuanto memoria requerida.

    Internamente el modulo almacena los datos como una estructura de ficheros
    Todo se puede almacenar como *datasets*, lo cual es semejante a `numpy.ndarrays`
    Pueden agruparse datasets en grupos.
    Para objetos pequenios, en vez de datasets pueden almacenarse atributos del
    dataset

    Referencia:
        * / : base
        * /miDataset : dataset
        * /miGrupo : grupo
        * /miGrupo/miDataset : dataset dentro del grupo
        * /miDataset.attrb['miAtributo']

    Los dataset son para almacenar objetos grandes (ej: numpy.ndarray, listas)
    Los atributos estan asociados a un dataset dado, y son mas eficientes
    para objetos mas pequeños (escalares y strings)


    Modos de apertura de los archivos:
        * r  Solo lectura, el archivo debe existir.
        * r+ Lectura y escritura, el archivo debe existir.
        * w  Crea el archivo, sobre-escribe si existe.
        * x  Crea el archivo, no sobrescribe.
        * a  Lectura y Escritura, Crea y sobreescribe.

    Note:
        El argumento *nuevo* debe utilizarse por unica vez, persistiendo la
        estructura de los datos, flotantes, listas, etc.

    Args:
        nombreArchivo (str): ruta + nombre del archivo a persistir los datos.
        valor (dict): datos a almacenar contenidos en un diccionario.
        nuevo (Opcional[bool]): Si es True, crea el archivo (sobreescribe).
            Utilizarlo para crear la estructura inicial.

    See Also:
        La documentacion oficial de `h5py`_.

        Adeḿas el la funcion :func:`cargarHDF5`.

    .. _h5py: http://docs.h5py.org/en/latest/quick.html

    Examples:
        >>> import numpy as np
        >>> datos = {'nombre': 'mlp', 'valor': 0.0, 'pesos': []}
        >>> a = np.asarray([[7,8,9],[10,11,12]])
        >>> guardarHDF5(nombreArchivo="prueba.cupdyle", valor=datos, nuevo=True)
        >>> datos
        {'nombre': 'mlp',
        'valor': 0.0,
        'pesos': []}
        >>> guardarHDF5("prueba.cupdyle",{'pesos':a})
        >>> guardarHDF5("prueba.cupdyle",{'valor':10.0, 'nombre':'mlp2'})
        >>> cargarHDF5("prueba.cupydle",None)
        {'nombre': 'mlp2',
        'valor': 10.0,
        'pesos': [array([7, 8, 9]), array([8, 9, 10])]}
    """
    if nuevo:
        h5f = h5py.File(nombreArchivo, 'w')
        h5f.create_dataset('atributos', data=0)
    else:
        h5f = h5py.File(nombreArchivo, 'r+')
        for k in valor.keys():
            assert k in h5f.keys() or k in h5f['atributos'].attrs

    for k in valor.keys():
        es_atributo=True
        if isinstance(valor[k],np.ndarray):
            if valor[k].shape[0]>1 or valor[k].shape[1]>1:
                es_atributo=False
        if isinstance(valor[k], list):
            es_atributo=False

        if es_atributo:
            #h5f['atributos'].attrs[k] = valor[k]
            h5f['atributos'].attrs[k] = downCast(valor[k])
            try:
                # si se guardo como atributo, trato de borrar si hay un dataset
                h5f.__delitem__(k)
            except:
                pass
                #h5f.close()
        else:
            if nuevo:
                if isinstance(valor[k], list):
                    h5f.create_group(k)
                else:
                    h5f.create_dataset(str(k), data=valor[k], dtype=np.float32)
            else:
                if isinstance(h5f[k], h5py.Group):
                    #if isinstance(valor[k], list):
                    # es como una lista, debo determinar el indice e incrementarlo dentro del grupo (lista)
                    dd=h5f[k].items()
                    import collections
                    h5f[k].create_dataset(k+str(len(collections.OrderedDict(sorted(dd)))),data=valor[k], dtype=np.float32)
                else:
                    h5f[k] = downCast(valor[k])
            try:
                # si se guardo como dataset, trato de borrar si hay un atributo
                h5f['atributos'].attrs.__delitem__(k)
            except:
                pass
                #h5f.close()
    h5f.close()
    return 0

def cargarHDF5(nombreArchivo, clave):
    """Recupera de disco el archivo almacenado con :func:`guardarHDF5`

    La estructura debe almacenarse previamente.

    Args:
        nombreArchivo (str): ruta + nombre del archivo a cargar los datos.
        clave (str, list(str), None): clave del objeto a cargar.
            Debe existir previamente. Si es una lista re recuperan los objetos.
            Si es `None` se recuperan todos los objetos en un diccionario.

    Return:
        dict, numpy.ndarray: Diccionario si clave=None o si clave=list(...).
        numpy.ndarray si clave=str

    See Also:
        La documentacion oficial de `h5py`_.

    .. _h5py: http://docs.h5py.org/en/latest/quick.html

        Adeḿas el la funcion :func:`guardarHDF5`.

    Examples:
        >>> import numpy as np
        >>> datos = {'nombre': 'mlp', 'valor': 0.0, 'pesos': []}
        >>> a = np.asarray([[7,8,9],[10,11,12]])
        >>> guardarHDF5(nombreArchivo="prueba.cupdyle", valor=datos, nuevo=True)
        >>> guardarHDF5("prueba.cupdyle",{'pesos':a})
        >>> guardarHDF5("prueba.cupdyle",{'valor':10.0, 'nombre':'mlp2'})
        >>> cargarHDF5("prueba.cupydle",'pesos')
        [array([7, 8, 9]), array([7, 8, 9]), array([ 8,  9, 10])]
        >>> cargarHDF5("prueba.cupydle",['nombre', 'pesos'])
        {'nombre': 'mlp',
        'pesos': [array([7, 8, 9]), array([7, 8, 9]), array([ 8,  9, 10])]}
        >>> cargarHDF5("prueba.cupydle",None)
        {'nombre': 'mlp2',
        'valor': 10.0,
        'pesos': [array([7, 8, 9]), array([7, 8, 9]), array([ 8,  9, 10])]}

    """
    datos = None

    h5f = h5py.File(nombreArchivo, 'r')

    if clave is not None:
        if isinstance(clave, list):
            datos = {}
            for k in clave:
                # busca primero si es un atributo
                if k in h5f["atributos"].attrs:
                    datos[k] = downCast(h5f["atributos"].attrs[k])
                elif isinstance(h5f[k], h5py.Group):    # busca si es un grupo (lista)
                    #datos[k] = []
                    #datos[k] = np.asarray(h5f[k])
                    # guarda una lista con los arrays contenidos en el grupo, ordenado
                    import collections
                    datos[k] = [np.asarray(val.value, dtype=np.float32) for key, val in collections.OrderedDict(sorted(h5f[k].items())).items()]
                else:
                    datos[k] = np.asarray(h5f[k], dtype=np.float32)
        else:
            # busca primero si es un atributo
            if clave in h5f["atributos"].attrs:
                datos = downCast(h5f["atributos"].attrs[clave])
            elif isinstance(h5f[clave], h5py.Group):    # busca si es un grupo (lista)
                # guarda una lista con los arrays contenidos en el grupo, ordenado
                import collections
                datos = [np.asarray(val.value, dtype=np.float32) for key, val in collections.OrderedDict(sorted(h5f[clave].items())).items()]
            else:
                datos = np.asarray(h5f[clave], dtype=np.float32)
    else: #clave es none, devuelvo todo, ojo
        datos = {}
        for k in h5f.keys():
            if k == 'atributos':
                for k in h5f['atributos'].attrs:
                    datos[k] = downCast(h5f['atributos'].attrs[k])
                    print("EL daot:", datos[k], type(datos[k]))
                    #assert False
            elif isinstance(h5f[k], h5py.Group):
                import collections
                datos[k] = [np.asarray(val.value, dtype=np.float32) for key, val in collections.OrderedDict(sorted(h5f[k].items())).items()]
            else:
                datos[k] = np.asarray(h5f[k], dtype=np.float32)
    h5f.close()
    return datos

def guardarSHELVE(nombreArchivo, valor, nuevo=False):
    """Metodo de almacenamiento por medio de SHELVE (pickles)

    Es menos eficiente que HDF5 con respecto a la memoria requerida, pero mas
    simple de posterior lectura

    Internamente almacena los datos como *pickles* individuales.

    Modos de apertura de los archivos:
        * r  Solo lectura, el archivo debe existir.
        * w  Lectura y escritura, el archivo debe existir.
        * c  Lectura y escritura, crea si no existe.
        * n  Lectura y Escritura, Crea y sobreescribe.

    Note:
        El argumento *nuevo* debe utilizarse por unica vez, persistiendo la
        estructura de los datos, flotantes, listas, etc.

    Args:
        nombreArchivo (str): ruta + nombre del archivo a persistir los datos.
        valor (dict): datos a almacenar contenidos en un diccionario.
        nuevo (Opcional[bool]): Si es True, crea el archivo (sobreescribe).
            Utilizarlo para crear la estructura inicial.

    See Also:
        La documentacion oficial de `shelve`_.

        Adeḿas el la funcion :func:`cargarSHELVE`.

    .. _shelve: https://docs.python.org/3.4/library/shelve.html

    Examples:
        >>> import numpy as np
        >>> datos = {'nombre': 'mlp', 'valor': 0.0, 'pesos': []}
        >>> a = np.asarray([[7,8,9],[10,11,12]])
        >>> guardarSHELVE(nombreArchivo="prueba.cupdyle", valor=datos, nuevo=True)
        >>> datos
        {'nombre': 'mlp',
        'valor': 0.0,
        'pesos': []}
        >>> guardarSHELVE("prueba.cupdyle",{'pesos':a})
        >>> guardarSHELVE("prueba.cupdyle",{'valor':10.0, 'nombre':'mlp2'})
        >>> cargarSHELVE("prueba.cupydle",None)
        {'nombre': 'mlp2',
        'valor': 10.0,
        'pesos': [array([7, 8, 9]), array([8, 9, 10])]}
    """
    assert False, "No contempla los mismos casos que HDF5"

    if nuevo:
        shelf = shelve.open(nombreArchivo, flag='n', writeback=False, protocol=2)
    else:
        shelf = shelve.open(nombreArchivo, flag='w', writeback=False, protocol=2)
        for k in valor.keys():
            assert k in shelf.keys(), "Clave no encontrada en el archivo"

    for key in valor.keys():
        if isinstance(valor[key] ,list):
            if nuevo:
                shelf[key] = valor[key]
            else: # ya hay guardada una lista
                tmp = shelf[key]
                tmp.extend(valor[key])
                shelf[key] = tmp
                del tmp
        else:
            shelf[key] = valor[key]
    shelf.close()
    return 0

def cargarSHELVE(nombreArchivo, clave):
    """Metodo para recuperar los datos almacenados por medio de *shelve*

    Es menos eficiente que HDF5 con respecto a la memoria requerida, pero mas
    simple de posterior lectura.

    Args:
        nombreArchivo (str): ruta + nombre del archivo a cargar los datos.
        clave (str, list(str), None): clave del objeto a cargar.
            Debe existir previamente. Si es una lista re recuperan los objetos.
            Si es `None` se recuperan todos los objetos en un diccionario.

    Return:
        dict, numpy.ndarray: Diccionario si clave=None o si clave=list(...).
        numpy.ndarray si clave=str

    See Also:
        La documentacion oficial de `shelve`_.

        Adeḿas el la funcion :func:`guardarSHELVE`.

    .. _shelve: https://docs.python.org/3.4/library/shelve.html


    Examples:
        >>> import numpy as np
        >>> datos = {'nombre': 'mlp', 'valor': 0.0, 'pesos': []}
        >>> a = np.asarray([[7,8,9],[10,11,12]])
        >>> guardarSHELVE(nombreArchivo="prueba.cupdyle", valor=datos, nuevo=True)
        >>> datos
        {'nombre': 'mlp',
        'valor': 0.0,
        'pesos': []}
        >>> guardarSHELVE("prueba.cupdyle",{'pesos':a})
        >>> guardarSHELVE("prueba.cupdyle",{'valor':10.0, 'nombre':'mlp2'})
        >>> cargarSHELVE("prueba.cupydle",None)
        {'nombre': 'mlp2',
        'valor': 10.0,
        'pesos': [array([7, 8, 9]), array([8, 9, 10])]}

    """
    datos = None
    with shelve.open(nombreArchivo, flag='r', writeback=False, protocol=2) as shelf:
        if clave is not None:
            assert clave in shelf.keys(), "clave no almacenada " + str(clave)
            datos = shelf[clave]
        else:
            datos = {key: shelf[key] for key in shelf.keys()}
        shelf.close()
    return datos

def save(objeto, filename=None, compression=None):
    assert filename is not None, "Nombre del archivo a guardar NO VALIDO"

    # en el filename esta la compression... la detecto
    if compression is None:
        if filename.find('.pkl') != -1: # quire decir que contiene la terminacion
            with open(filename, "wb") as f:
                pickle.dump(objeto, f)
                f.close()
        elif filename.find('.pgz') != -1: # quire decir que contiene la terminacion
            with gzip.GzipFile(filename, "w") as f:
                pickle.dump(objeto, f)
                f.close()
        elif filename.find('.pbz2') != -1:
            with bz2.BZ2File(filename, 'w') as f:
                pickle.dump(objeto, f)
                f.close()
        elif filename.find('.zip') != -1:
            with gzip.GzipFile(filename, 'w') as f:
                pickle.dump(objeto, f)
                f.close()
        else:
            try:
                with open(filename + '.pkl', "wb") as f:
                    pickle.dump(objeto, f)
                    f.close()
            except:
                print("Archivo no encontrado", filename)
                sys.exit(0)

    elif compression == 'gzip':
        with gzip.GzipFile(filename + '.pgz', 'w') as f:
            pickle.dump(objeto, f)
            f.close()
    elif compression == 'bzip2':
        with bz2.BZ2File(filename + '.pbz2', 'w') as f:
            pickle.dump(objeto, f)
            f.close()
    elif compression == 'zip':
        with gzip.GzipFile(filename + '.zip', 'wb') as f:
            pickle.dump(objeto,f)
            f.close()
    else:
        sys.exit("Parametro de compresion no se reconoce")
    return objeto

def load(filename=None, compression=None):
    assert filename is not None, "Nombre del archivo NO VALIDO"
    objeto = None

    if compression is None:
        if filename.find('.pkl') != -1: # quire decir que contiene la terminacion
            with open(filename, "rb") as f:
                objeto = pickle.load(f)
                f.close()
        elif filename.find('.pgz') != -1: # quire decir que contiene la terminacion
            with gzip.open(filename, "rb") as f:
                objeto = pickle.load(f)
                f.close()
        elif filename.find('.pbz2') != -1:
            with bz2.open(filename, 'rb') as f:
                objeto = pickle.load(f)
                f.close()
        else:
            try:
                with open(filename + '.pkl', "rb") as f:
                    objeto = pickle.load(f)
                    f.close()
            except:
                print("Archivo no encontrado", filename)
                sys.exit(0)
    elif compression == 'gzip':
        with gzip.open(filename + '.pgz', "rb") as f:
            objeto = pickle.load(f)
            f.close()
    elif compression == 'bzip2':
        with bz2.open(filename + '.pbz2', 'rb') as f:
            objeto = pickle.load(f)
            f.close()
    elif compression == 'zip':
        with gzip.open(filename + '.zip', 'rb') as f:
            objeto = pickle.load(f)
            f.close()
    else:
        sys.exit("Parametro de compresion no se reconoce")

    return objeto

class temporizador(object):
    '''
    tipo singleton
    https://es.wikipedia.org/wiki/Singleton#Python
    '''

    instance = None
    inicio = None
    fin = None

    def __new__(cls, *args, **kargs):
        if cls.instance is None:
            cls.instance = object.__new__(cls, *args, **kargs)
        return cls.instance

    def tiempo(self):
        return timeit.default_timer()

    @staticmethod
    def time():
        return timeit.default_timer()

    def tic(cls):
        '''
        retorna una marca del temporizador...
        se ejecuta dos veces, una al inicio y otra al final...
        asi se cuenta la diferencia
        '''
        # realiza un tic
        cls.inicio = cls.tiempo()
        cls.fin = None # lo borro
        return cls.inicio


    def toc(cls):
        '''
        '''
        cls.fin = cls.tiempo()
        if cls.inicio is None:
            cls.inicio = cls.fin
        return cls.fin

    # todo arreglar
    def transcurrido(self, start=inicio, end=fin, string=True):
        if start is None:
            return ""
        if end is None:
            end = self.toc()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        if string:
            tiempo = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        else:
            tiempo = abs(end - start)
        return tiempo


def safe_indexing(X, indices):
    """Return items or rows from X using indices.
    Allows simple indexing of lists or arrays.
    Parameters
    ----------
    X : array-like, sparse-matrix, list.
        Data from which to sample rows or items.
    indices : array-like, list
        Indices according to which X will be subsampled.
    """
    if hasattr(X, "iloc"):
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def indexable(*iterables):
    """Make arrays indexable for cross-validation.
    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.
    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if sp.issparse(X):
            result.append(X.tocsr())
        elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    check_consistent_length(*result)
    return result

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class RestrictedDict(dict):
    """Es un diccionario, cualquier intento de almacenar una clave ya existente
    se arroja un error.

    Se inicializa con una instancia, y las claves permitidas que no pueden ser modificadas.

    Args:
        allowed_keys (tuple): claves permidas

    Examples:
        >>> p = RestrictedDict(('x','y'))
        >>> print p
        RestrictedDict(('x', 'y'), {})
        >>> p['x'] = 1
        >>> p['y'] = 'item'
        >>> print p
        RestrictedDict(('x', 'y'), {'y': 'item', 'x': 1})
        >>> p.update({'x': 2, 'y': 5})
        >>> print p
        RestrictedDict(('x', 'y'), {'y': 5, 'x': 2})
        >>> p['x']
        2
        >>> p['z'] = 0
        Traceback (most recent call last):
        ...
        KeyError: 'z is not allowed as key'
        >>> q = RestrictedDict(('x', 'y'), x=2, y=5)
        >>> p==q
        True
        >>> q = RestrictedDict(('x', 'y', 'z'), x=2, y=5)
        >>> p==q
        False
        >>> len(q)
        2
        >>> q.keys()
        ['y', 'x']
        >>> q._allowed_keys
        ('x', 'y', 'z')
        >>> p._allowed_keys = ('x', 'y', 'z')
        >>> p['z'] = 3
        >>> print p
        RestrictedDict(('x', 'y', 'z'), {'y': 5, 'x': 2, 'z': 3})

    """

    def __init__(self, allowed_keys, seq=(), **kwargs):
        super(RestrictedDict, self).__init__()
        self._allowed_keys = tuple(allowed_keys)
        # normalize arguments to a (key, value) iterable
        if hasattr(seq, 'keys'):
            get = seq.__getitem__
            seq = ((k, get(k)) for k in seq.keys())
        if kwargs:
            from itertools import chain
            seq = chain(seq, kwargs.iteritems())
        # scan the items keeping track of the keys' order
        for k, v in seq:
            self.__setitem__(k, v)

    def __setitem__(self, key, value):
        """Checks if the key is allowed before setting the value"""
        if key in self._allowed_keys:
            super(RestrictedDict, self).__setitem__(key, value)
        else:
            raise KeyError("'{}' no es una key permitida".format(key))

    def update(self, e=None, **kwargs):
        """
        Equivalent to dict.update(), but it was needed to call
        RestrictedDict.__setitem__() instead of dict.__setitem__
        """
        try:
            for k in e:
                self.__setitem__(k, e[k])
        except AttributeError:
            for (k, v) in e:
                self.__setitem__(k, v)
        for k in kwargs:
            self.__setitem__(k, kwargs[k])

    def __eq__(self, other):
        """
        Two RestrictedDicts are equal when their dictionaries and allowed keys
        are all equal.
        """
        if other is None:
            return False
        try:
            allowedcmp = (self._allowed_keys == other._allowed_keys)
            if allowedcmp:
                dictcmp = super(RestrictedDict, self).__eq__(other)
            else:
                return False
        except AttributeError:
            #Other is not a RestrictedDict
            return False
        return bool(dictcmp)

    def __ne__(self, other):
        """x.__ne__(y) <==> not x.__eq__(y)"""
        return not self.__eq__(other)

    def __repr__(self):
        """Representation of the RestrictedDict"""
        return 'RestrictedDict(%s, %s)' % (self._allowed_keys.__repr__(),
                                     super(RestrictedDict, self).__repr__())

def z_score(datos):
    # Normalizacion estadistica (Z-score Normalization)
    # https://en.wikipedia.org/wiki/Standard_score
    # datos [N x D] (N # de datos, D su dimensionalidad).
    # x' = (x - mu)/sigma
    # x = dato de entrada; x' = dato normalizado
    # mu = media; sigma = desvio estandar
    mu = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    return (datos - mu) / sigma

def min_max_escalado(datos, min_obj=0.0, max_obj=1.0):
    # Normalizacion Min-Max
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    # x' = (x-min)/(max-min)*(max_obj-min_obj)+min_obj
    # x = datos de entrada; x' = dato normalizado
    # max_obj = limite superior del rango objetivo
    # min_obj = limite inferior del rango objetivo
    minimo = np.min(datos)
    maximo = np.max(datos)
    x = ((datos - minimo) / (maximo - minimo)) * ( max_obj - min_obj) + min_obj
    return x

def blanqueo(datos, eigenvalues=100, epison=1e-5):
    # whitening PCA normalization
    # https://en.wikipedia.org/wiki/Whitening_transformation
    # http://cs231n.github.io/neural-networks-2/
    # Assume input data matrix X of size [N x D]
    datos -= np.mean(datos, axis = 0) # zero-center the data (important)
    cov = np.dot(datos.T, datos) / datos.shape[0] # get the data covariance matrix
    U,S,V = np.linalg.svd(cov)
    datos_rot = np.dot(datos, U) # decorrelate the data
    # similar a PCA...
    # se quedan con los eigenvalues (dimensiones) mas importantes (contienen mas varianza)
    datos_rot_reduced = np.dot(datos, U[:,:eigenvalues]) # Xrot_reduced becomes [N x 100]
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    datos_white = datos_rot / np.sqrt(S + epison)
    return datos_white
