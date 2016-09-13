#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

import sys
import pickle
import bz2 # bzip2
import gzip # gzip

import timeit

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
