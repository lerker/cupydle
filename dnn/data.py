__author__ = 'lerker'
import random
import numpy as np
import cupydle.dnn.fileio as fileio


class LabeledDataSet(object):
    def __init__(self, path, delimiter=';'):
        self.data = fileio.load_file(path, separador=delimiter)

    def split_data(self, labelRow=-1):
        entrada = np.array(self.data[:, 0:labelRow - 1], dtype=np.dtype(float))
        salida = np.array(self.data[:, labelRow - 1:], dtype=np.dtype(float))
        return entrada, salida


class LabeledDataSet2(object):
    def __init__(self, data=None):
        self.with_lp = False  # Flag que indica si se almaceno entradas en forma de LabeledPoints
        self.data = data
        if self.with_lp is False and self.data is not None:  # Por ahora que quede etiquetado con LabeledPoints
            self.labeled_point()

    # TODO ver si conviene que sean properties
    @property
    def features(self):
        if self.with_lp is True:
            features = self.data.map(lambda lp: lp.features)
        else:
            # TODO
            1
            # features = self.data.map(lambda (l, f): f)
        return features

    @property
    def labels(self):
        if self.with_lp is True:
            labels = self.data.map(lambda lp: lp.label)
        else:
            labels = 0
            # labels = self.data.map(lambda (l, f): l)
        return labels

    def load_file(self, path):
        self.data = fileio.load_file(path)
        self.labeled_point()

    def save_file(self, path):  # TODO mejorar pq no anda
        fileio.save_file(self.data, path)

    def labeled_point(self):
        if self.with_lp is False:
            if self.data is not None:
                self.data = 0
                # self.data = self.data.map(lambda (l, f): LabeledPoint(l, f))
            self.with_lp = True
        else:
            if self.data is not None:
                self.data = self.data.map(lambda lp: (lp.label, lp.features))
            self.with_lp = False

    def split_data(self, fractions, seed=123):
        sets = split_data(self.data, fractions, seed)
        sets = [LabeledDataSet(data) for data in sets]
        return sets

    @property
    def shape(self):
        shape = None
        if self.data is not None:
            rows = self.data.count()
            cols = len(self.features.take(1)[0].toArray())
            shape = (rows, cols)
        return shape


def label_to_vector(label, n_classes):
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)


# TODO: hacer sample ponderando las clases por error en validacion
def subsample(data, size, balanced=True, seed=123):
    """
    Muestreo de data, con resultado balanceado por clases si se lo pide
    :param data: list of LabeledPoint
    :param size: int
    :param seed: int
    :return:

    """
    random.seed(seed)
    if balanced is True:  # Problema de clasificacion
        n_classes = int(max(map(lambda lp: lp.label, data))) + 1
        size = size / n_classes  # es un int, y puede resultar menor al ingresado (se trunca)
        sample = []
        for c in range(n_classes):
            batch_class = filter(lambda lp: lp.label == c, data)  # Filtro entradas que pertenezcan a la clase c
            batch = random.sample(batch_class, size)
            sample.extend(batch)  # Agrego el batch al vector de muestreo
    else:  # Problema de regresion
        sample = random.sample(data, size)
    random.shuffle(sample)  # Mezclo para agregar aleatoriedad
    return sample


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

    # Mezclo un poco los datos
    random.seed(seed)
    random.shuffle(data)
    # Segmento conjuntos
    size_data = len(data)
    size_split = map(lambda f: int(size_data * f), fractions)
    index_split = [0] + size_split[:-1]
    sets = [data[i:i + size] for i, size in zip(index_split, size_split)]
    return sets


def label_data(data, label):
    labeled_data = 0
    # labeled_data = map(lambda (x, y): LabeledPoint(y, x), zip(data, label))
    return labeled_data
