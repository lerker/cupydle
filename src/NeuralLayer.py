import numpy as np

import activations as act
from Neurons import Neurons as Neurons

__author__ = "Nelson Ponzoni"
__copyright__ = "Copyright 2015-2016, Proyecto Final de Carrera"
__credits__ = ["Nelson Ponzoni"]
__license__ = "GPL"
__version__ = "20160101"
__maintainer__ = "Nelson Ponzoni"
__email__ = "npcuadra@gmail.com"
__status__ = "Production"

"""

"""


class NeuralLayer(object):
    def __init__(self,
                 n_in=2,
                 n_out=2,
                 activation='Sigmoid',
                 w=None,
                 b=None,
                 rng=None):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = act.activation_functions[activation]  # activacion de la capa
        self.activation_d = act.activation_functions_prime[activation]  # derivada de la activacion de la capa

        # no le di semilla?
        if rng is None:
            rng = np.random.seed(1234)

        self.rng = rng
        self.shape_w = n_out, n_in  # tupla
        self.shape_b = n_out, 1

        # si los pesos no fueron pasados, es porque hay que inicializarlos
        if w is None:
            # inicializacion media rara de deeplearning.com
            w = np.asarray(
                np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                                  high=+np.sqrt(6.0 / (n_in + n_out)),
                                  size=self.shape_w),
                dtype=np.dtype(float))
        if b is None:
            b = np.zeros(self.shape_b, dtype=np.dtype(float))

        self.weights = Neurons(w, self.shape_w)
        self.bias = Neurons(b, self.shape_b)

    # propiedades intrisecas de las capas
    def __str__(self):
        return str("W:\n" + str(self.weights) + "\nbias:\n" + str(self.bias))

    def __mul__(self, other):
        self.weights *= other
        self.bias *= other
        return self

    # funciones para obtener valores
    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def output(self, x, grad=False):
        """
        Activacion de una capa con una entrada, si pido el gradiente lo retorno como una tupla
        :param x: entrada a la capa, debe ser una neurona
        :param grad:
        :return:
        """
        # multiplicacion de los pesos por la entrada
        wx = self.weights.mul_array(x)
        # adicion del bias
        z = wx.sum_array(self.bias)
        # a la salida lineal, es pasada por la funcion de activacion de transferencia
        a = z.activation(self.activation)
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)

        return a


    def update(self, step_w, step_b):
        # Actualiza sumando los argumentos w y b a los respectivos pesos
        self.weights += step_w
        # self.weights_T += step_w.transpose()
        self.bias += step_b
        return

    def outputLinear(self, x):
        """
        Activacion de una capa con una entrada, si pido el gradiente lo retorno como una tupla
        :param x: entrada a la capa, debe ser una neurona
        :param grad:
        :return:
        """
        # multiplicacion de los pesos por la entrada
        wx = self.weights.mul_array(x)
        # adicion del bias
        z = wx.sum_array(self.bias)
        # a la salida lineal, es pasada por la funcion de activacion de transferencia
        return z



class ClassificationLayer(NeuralLayer):
    def output(self, x, grad=False):
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.softmax()  # La activacion es un clasificador softmax
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
        return a