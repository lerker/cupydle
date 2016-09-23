#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

import sys
import pickle
import bz2 # bzip2
import gzip # gzip

import timeit

import numbers

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"


def mostrar_tamaniode(x, level=0):
    # imprime el tamanio de cualquier objeto pasado
    print("\t" * level, x.__class__, sys.getsizeof(x), x)

    if hasattr(x, '__iter__'):
        if hasattr(x, 'items'):
            for xx in x.items():
                mostrar_tamaniode(xx, level + 1)
        else:
            for xx in x:
                mostrar_tamaniode(xx, level + 1)
    return 0


def subsample(data, size, balanced=True, seed=123):
    """

    :param data: datos a extraer un batch (ya contiene el resultado)
    :param size: tamanio del mini_bach
    :param balanced:
    :param seed:
    :return:
    """
    return data[np.random.choice(data.shape[0], size=(size,))]

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
    def transcurrido(self, start=inicio, end=fin):
        if start is None:
            return ""
        if end is None:
            end = self.toc()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        tiempo = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
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
    """
    Stores the properties of an object. It's a dictionary that's
    restricted to a tuple of allowed keys. Any attempt to set an invalid
    key raises an error.

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
        """
        Initializes the class instance. The allowed_keys tuple is
        required, and it cannot be changed later.
        If seq and/or kwargs are provided, the values are added (just
        like a normal dictionary).
        """
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
