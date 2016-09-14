#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"


"""

"""

import numpy
import time
import os
import sys
import subprocess
import argparse

# Dependencias Externas
## Core
from cupydle.dnn.funciones import sigmoideaTheano
## Data
from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk
## Utils
from cupydle.dnn.utils import temporizador

from cupydle.dnn.mlp import MLP

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de un MLP sobre FACE.')
    parser.add_argument('-d', '--directorio',   type=str,   dest="directorio",  default='test_MLP', help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',            type=str,   dest="dataset",     default='videos_clases_procesados_zscore_minmax.npz', help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-l', '--capas',        type=int,   dest="capas",       required=True,      nargs='+',          help="Capas del MLP, #entrada, #oculta1, .... , #salida")
    parser.add_argument('-b', '--batchsize',    type=int,   dest="tambatch",    default=10,         required=False,     help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-e', '--epocas',       type=int,   dest="epocas",      default=10,         required=False,     help="Cantidad de epocas")
    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio_ejecucion  = argumentos.directorio
    capas                 = argumentos.capas
    dataset               = argumentos.dataset
    tambatch              = argumentos.tambatch
    epocas                = argumentos.epocas


    # configuraciones con respecto a los directorios
    directorioActual= os.getcwd()                                  # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest     = directorio_ejecucion + '/'                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)

    if not os.path.exists(rutaDatos):
        print("Creando la base de datos en:", rutaDatos)
        os.makedirs(rutaDatos)

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos archivos, 'videos' y 'clases'
    b = numpy.load(rutaDatos + dataset)
    videos = b['videos']
    clases = b['clases']
    del b #libera memoria

    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...
    datos = []
    datosTRN = (videos, clases)
    datosVAL = (videos[100:200,0:],clases[100:200])
    datosTST = (videos[200:300,0:],clases[200:300:])

    datos.append(datosTRN); datos.append(datosVAL); datos.append(datosTST)

    # creo la red
    clasificador = MLP(clasificacion=True,
                       rng=None,
                       ruta=rutaCompleta)

    # se agregan los parametros
    clasificador.setParametroEntrenamiento({'tasaAprendizaje':0.01})
    clasificador.setParametroEntrenamiento({'regularizadorL1':0.00})
    clasificador.setParametroEntrenamiento({'regularizadorL2':0.0001})
    clasificador.setParametroEntrenamiento({'momento':0.0})
    clasificador.setParametroEntrenamiento({'activationfuntion':sigmoideaTheano()})
    clasificador.setParametroEntrenamiento({'epocas':epocas})
    clasificador.setParametroEntrenamiento({'toleranciaError':0.08})

    # agrego tantas capas desee de regresion
    # la primera es la que indica la entrada
    # las intermedias son de regresion
    # la ultima es de clasificaicon
    clasificador.agregarCapa(unidadesEntrada=capas[0], unidadesSalida=capas[1], clasificacion=False, activacion=sigmoideaTheano(), pesos=None, biases=None)
    for idx, capa in enumerate(capas[1:-1]):
        clasificador.agregarCapa(unidadesSalida=capa, clasificacion=False, activacion=sigmoideaTheano(), pesos=None, biases=None)
    clasificador.agregarCapa(unidadesSalida=capas[-1], clasificacion=True, pesos=None, biases=None)

    T = temporizador()
    inicio = T.tic()

    # se entrena la red
    clasificador.entrenar(trainSet=datos[0],
                          validSet=datos[1],
                          testSet=datos[2],
                          batch_size=tambatch)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    # guardando los parametros aprendidos
    clasificador.guardarParametros()

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
