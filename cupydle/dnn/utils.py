import numpy as np
import random

__author__ = 'lerker'


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