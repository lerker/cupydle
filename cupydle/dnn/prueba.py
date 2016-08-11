import numpy
import os


directorioActual= os.getcwd()                                   # directorio actual de ejecucion
rutaTest        = directorioActual + '/cupydle/test/mnist/'     # sobre el de ejecucion la ruta a los tests
rutaDatos       = directorioActual + '/cupydle/data/DB_mnist/'  # donde se almacenan la base de datos
carpetaTest     = 'test_RBM/'                                   # carpeta a crear para los tests
rutaCompleta    = rutaTest + carpetaTest

pesos1 = numpy.load(fullPath + "pesos1.npy")

