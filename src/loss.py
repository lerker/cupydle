import numpy as np

__author__ = "Nelson Ponzoni"
__copyright__ = "Copyright 2015-2016, Proyecto Final de Carrera"
__credits__ = ["Nelson Ponzoni"]
__license__ = "GPL"
__version__ = "20160101"
__maintainer__ = "Nelson Ponzoni"
__email__ = "npcuadra@gmail.com"
__status__ = "Production"

"""
Loss funtions to evaluate de error, difference of real to expected value
"""


def mse(value, target):
    err = target - value
    n = err.size
    return np.sum(np.square(err)) / (1.0 * n)


def mse_prime(value, target):
    err = target - value
    n = err.size
    return 2 * err / (1.0 * n)


def resta(self, y):
    n = self.count
    error = (y - self.matrix)
    sum_error = sum(error)
    return sum_error / (1.0 * n)

def cross_entropy(y, t):
    num_classes = len(y)
    t = label_to_vector(t, num_classes)
    return -sum(t * np.log(y))[0]


def cross_entropy_d(y, t):
    num_classes = len(y)
    t = label_to_vector(t, num_classes)
    return y - t


def label_to_vector(label, n_classes):
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)

fun_loss = {'MSE': mse, 'resta': resta, 'CROSS_ENTROPY': cross_entropy}
fun_loss_prime = {'MSE': mse_prime, 'CROSS_ENTROPY': cross_entropy_d}
