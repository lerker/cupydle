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
Activations functions, they will be used for 'activate' the signal of de neuron, the linear signal is passed throw the
activation function and loss the linearity.
"""


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

# dictionaries to do the job more easy...
activation_functions = {'Tanh': tanh, 'Sigmoid': sigmoid}
activation_functions_prime = {'Tanh': tanh_prime, 'Sigmoid': sigmoid_prime}

"""
x = np.arange(-10, 10, 0.1)
import matplotlib.pyplot as plt

#plt.plot(x, sigmoid(x))
plt.plot(x, tanh(x))
plt.show()
"""
