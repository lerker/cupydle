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
from cupydle.dnn.unidades import UnidadBinaria
from cupydle.dnn.unidades import UnidadGaussiana
## Data
from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk
## Utils
from cupydle.dnn.utils import temporizador

from cupydle.dnn.rbm import RBM


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre FACE')
    parser.add_argument('-d', '--directorio',   type=str,   dest="directorio",  default='test_RBM', help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',            type=str,   dest="dataset",     default='videos_clases_procesados_zscore_minmax.npz', help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-b', '--batchsize',    type=int,   dest="tambatch",    default=10,         required=False,     help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-e', '--epocas',       type=int,   dest="epocas",      default=10,         required=False,     help="Cantidad de epocas")
    parser.add_argument('-p', '--porcentaje',   type=float, dest="porcentaje",  default=0.8,        required=False,     help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-v', '--visibles',     type=int,   dest="visibles",    default=230300,     required=False,     help="Cantidad de Unidades Visibles")
    parser.add_argument('-o', '--ocultas',      type=int,   dest="ocultas",     default=100,        required=False,     help="Cantidad de Unidades Ocultas")
    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio_ejecucion  = argumentos.directorio
    dataset               = argumentos.dataset
    tambatch              = argumentos.tambatch
    epocas                = argumentos.epocas
    porcentaje            = argumentos.porcentaje
    visibles              = argumentos.visibles
    ocultas               = argumentos.ocultas

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
    # las clases estan desde 1..6, deben ser desde 0..5
    clases -= 1

    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...
    cantidad = int(clases.shape[0] * porcentaje)
    # la cantidad de ejemplos debe ser X partes enteras del minibatch, para que el algoritmo tome de a cachos y procese
    # es por ello que acomodo la cantidad hasta que quepa
    while (cantidad % tambatch):
        cantidad += 1
        assert cantidad != clases.shape[0], "Porcentaje trn/test muy alto, disminuir"

    datos = []
    datos = [(videos[:cantidad], clases[:cantidad]), (videos[cantidad:],clases[cantidad:])]

    # creo la red
    red = RBM(n_visible=visibles, n_hidden=ocultas, ruta=rutaCompleta)

    parametros={'epsilonw':0.1,
                'epsilonvb':0.1,
                'epsilonhb':0.1,
                'momentum':0.0,
                'weightcost':0.0,
                'unidadesVisibles':UnidadBinaria(),
                'unidadesOcultas':UnidadBinaria(),
                'dropoutVisibles': 1.0, # probabilidad de actividad en la neurona, =1 todas, =0 ninguna
                'dropoutOcultas': 1.0} # probabilidad de actividad en la neurona, =1 todas, =0 ninguna

    red.setParams(parametros)
    red.setParams({'epocas':epocas})

    T = temporizador()
    inicio = T.tic()

    red.entrenamiento(data=datos[0][0],
                      tamMiniBatch=tambatch,
                      tamMacroBatch=None,
                      pcd=False,
                      gibbsSteps=1,
                      validationData=None,
                      filtros=True)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    print('Guardando el modelo en ...', rutaCompleta)
    inicio = T.tic()
    red.guardar(nombreArchivo="rbm_face.zip")
    final = T.toc()
    print("Tiempo total para guardar: {}".format(T.transcurrido(inicio, final)))

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
