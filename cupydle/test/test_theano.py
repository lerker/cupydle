""" Use this file to check the speed difference between a big matrix multiplication
performed on the GPU or on the CPU.
"""

import theano
import theano.tensor as T
from theano import function, shared
import numpy as np


import time
x = T.matrix('x', dtype=theano.config.floatX)
y = T.matrix('y', dtype=theano.config.floatX)

sc = shared(np.zeros((10, 10), dtype = theano.config.floatX), name='sc')

mydot = function( [x,y], updates=( (sc, T.dot(x,y)), ))

# We need to declare the variables shared to run on GPU
a = np.ones((1000, 1000), dtype = theano.config.floatX) * 40.0
b = np.ones((1000, 1000), dtype = theano.config.floatX) * 23.0
print("go")

before = time.time()
mydot(a,b)
print(time.time() - before)

print(sc.get_value().sum())


"""
en el servidor

http://stackoverflow.com/questions/8917977/installing-lapack-for-numpy



nponzoni@gauss:~$ sudo apt-get remove libopenblas-base
Leyendo lista de paquetes... Hecho
Creando árbol de dependencias
Leyendo la información de estado... Hecho
Los siguientes paquetes se ELIMINARÁN:
  libopenblas-base libopenblas-dev
0 actualizados, 0 se instalarán, 2 para eliminar y 22 no actualizados.
Se liberarán 50,7 MB después de esta operación.
¿Desea continuar [S/n]?
(Leyendo la base de datos ... 360108 ficheros o directorios instalados actualmente.)
Desinstalando libopenblas-dev ...
update-alternatives: utilizando /usr/lib/atlas-base/atlas/libblas.so para proveer /usr/lib/libblas.so (libblas.so) en modo automático.
Desinstalando libopenblas-base ...
update-alternatives: utilizando /usr/lib/atlas-base/atlas/libblas.so.3gf para proveer /usr/lib/libblas.so.3gf (libblas.so.3gf) en modo automático.
nponzoni@gauss:~$ sudo pip install Theano



http://blog.nguyenvq.com/blog/2014/11/10/optimized-r-and-python-standard-blas-vs-atlas-vs-openblas-vs-mkl/
"""
