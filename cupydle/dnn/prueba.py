import numpy
import os


directorioActual= os.getcwd()                                   # directorio actual de ejecucion
rutaTest        = directorioActual + '/cupydle/test/mnist/'     # sobre el de ejecucion la ruta a los tests
rutaDatos       = directorioActual + '/cupydle/data/DB_mnist/'  # donde se almacenan la base de datos
carpetaTest     = 'test_RBM/'                                   # carpeta a crear para los tests
rutaCompleta    = rutaTest + carpetaTest

#pesos1 = numpy.load(fullPath + "pesos1.npy")

class MLP(object):
    verbose=True

    def coso(self):
        print("es verdadero") if MLP.verbose else None
        return


from numpy.random import RandomState as npRandom

print(npRandom(1234))
h=True
cosa = (0 if h is None else 1)
print(cosa)
cosito = MLP()
cosito2 = MLP()

cosito.coso()
MLP.verbose=False
cosito.coso()

def check():
    return 1

if check():
    print('Entre')

class Foo(object):
  def __init__(self, val=2):
     self.val = val
  def __getstate__(self):
     print("I'm being pickled")
     self.val *= 2
     return self.__dict__
  def __setstate__(self, d):
     print("I'm being unpickled with these values:", d)
     self.__dict__ = d
     self.val *= 3

import pickle
f = Foo()
f_string = pickle.dumps(f, protocol=pickle.HIGHEST_PROTOCOL)
f_new = pickle.loads(f_string)
print(f.val)
print(f_string)


# Python 3.4+
from abc import ABC, abstractmethod
class animal(ABC):

    def __init__(self):
        self.variable = 10

    def cambiar(self):
        self.variable=15

    @abstractmethod
    def coso(self):
        return "soy un animal"

    def vuelo(self):
        return "yo vuelo"

class perro(animal):
    def caca(self):
        return "hago caca"
    def coso(self):
        h = super(perro, self).coso()
        super(perro, self).cambiar()

        return h + " soy un perro"

a = perro()
print(a.coso())
print(a.variable)


class coso(object):
    def __call__(self, x):
        return x*2

class coso2(object):
    def coso22(self, x):
        return coso(x)
o=coso()
print(o(2))


from cupydle.dnn.funciones import sigmoideaTheano


a = sigmoideaTheano()
a.dibujar()
