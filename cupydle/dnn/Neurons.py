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
Neurons class, this is abstraction of various neurons, a pack of neurons, that compose a neural layer
"""

class Neurons(object):
    def __init__(self, mat, shape):
        if len(shape) == 1:
            shape += (1,)
        if isinstance(mat, list):
            mat = np.array(mat)
        # la matriz debe tener la forma deseada
        self.matrix = mat.reshape(shape)
        self.rows = shape[0]
        self.cols = shape[1]

    # propiedades intrisecas de las neuronas
    @property
    def shape(self):
        return self.rows, self.cols

    @property
    def count(self):
        rows, cols = self.shape
        return rows * cols

    def __str__(self):
        return str(self.matrix)

    def __mul__(self, other):
        # TODO esta linea puede que este al pedo
        if isinstance(other, Neurons):
            other = other.matrix
        return Neurons(self.matrix * other, self.shape)

    def __div__(self, other):
        if isinstance(other, Neurons):
            other = other.matrix
        return Neurons(self.matrix / other, self.shape)

    def __sub__(self, other):
        if isinstance(other, Neurons):
            other = other.matrix
        return Neurons(self.matrix - other, self.shape)

    def __add__(self, other):
        if isinstance(other, Neurons):
            other = other.matrix
        return Neurons(self.matrix + other, self.shape)

    def __pow__(self, power, modulo=None):
        return Neurons(self.matrix ** power, self.shape)

    # opraciones basicas
    def mul_elemwise(self, array):
        if isinstance(array, Neurons):
            array = array.matrix
        return Neurons(np.multiply(self.matrix, array), self.shape)

    def mul_array(self, array):
        if isinstance(array, Neurons):
            array = array.matrix
        arrshape = array.shape
        if len(arrshape) == 1:
            arrshape += (1,)  # Le agrego la dimension que le falta
        shape = self.rows, arrshape[1]
        return Neurons(self.matrix.dot(array), shape)

    def sum_array(self, array):
        if isinstance(array, Neurons):
            array = array.matrix
        return Neurons(self.matrix + array, self.shape)

    def dot(self, vec):
        return self.matrix.dot(vec)

    def outer(self, array):
        if isinstance(array, Neurons):
            array = array.matrix
        res = np.outer(self.matrix, array)
        shape = res.shape
        return Neurons(res, shape)

    def transpose(self):
        return Neurons(self.matrix.transpose(), self.shape[::-1])

    def mse(self, y):
        n = self.count
        error = (y - self.matrix) ** 2
        sum_error = sum(error)
        return sum_error / (1.0 * n)

    def resta(self, y):
        n = self.count
        error = (y - self.matrix)
        sum_error = sum(error)
        return sum_error / (1.0 * n)


    def loss(self, fun, y):
        return fun(self.matrix, y)

    def loss_d(self, fun, y):
        return Neurons(fun(self.matrix, y), self.shape)

    def activation(self, fun):
        # el map no anda mas en python3 como iterador,
        # ver: http://stackoverflow.com/questions/28524378/convert-map-object-to-numpy-array-in-python-3
        return Neurons(list(map(lambda x: fun(x), self.matrix)), self.shape)

    def softmax(self):
        # Uso de tip de implementacion (http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)
        x = self.matrix
        x = x - max(x)  # Se previene valores muy grandes del exp con valores altos de x
        softmat = np.exp(x) / (sum(np.exp(x)))
        return Neurons(softmat, self.shape)
