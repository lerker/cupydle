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

from cupydle.dnn.utils import vectorize_label


def mse(value, target):
    err = target - value
    n = err.size
    return np.sum(np.square(err)) / (1.0 * n)


def mse_prime(value, target):
    err = target - value
    n = err.size
    return 2 * err / (1.0 * n)


def cross_entropy(a, y):
    """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
    return np.sum(np.nan_to_num(-1.0 * y * np.log(a) - (1 - y) * np.log(1 - a)))


def cross_entropy_d(value, target):
    # http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    # value<->target: deberia ser al reves me parece
    value = vectorize_label(label=value[0], n_classes=len(target))
    return target - value


def label_to_vector(label, n_classes):
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)

fun_loss = {'MSE': mse, 'CROSS_ENTROPY': cross_entropy}
fun_loss_prime = {'MSE': mse_prime, 'CROSS_ENTROPY': cross_entropy_d}
