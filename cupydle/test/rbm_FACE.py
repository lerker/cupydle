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


from numpy import genfromtxt


if __name__ == "__main__":

    directorioActual= os.getcwd()                                   # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest     = 'face_rbm/'                                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)

    if not os.path.exists(rutaDatos):
        print("Creando la base de datos en:", rutaDatos)
        os.makedirs(rutaDatos)


    setName = "face"

    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre FACE.')
    parser.add_argument('-g', '--guardar', action="store_true", dest="guardar", help="desea guardar (correr desde cero)", default=False)
    parser.add_argument('-m', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    guardar = args.guardar
    modelName = args.modelo
    #modelName = 'capa1.pgz'


    # obtengo todos los subconjuntos
    #train_img,  train_labels= mn.get_training()
    #test_img,   test_labels = mn.get_testing()
    #val_img,    val_labels  = mn.get_validation()

    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...

    #datos = [mn.get_training(), mn.get_testing(), mn.get_validation()]
    #datos = [( ((x/255.0).astype(numpy.float32)), y) for x, y in datos]

    b = numpy.load(rutaDatos + 'videos_clases.npz')

    videos = b['videos']
    clases = b['clases']

    datos = [(videos[0:20,0:], clases[0:20]), (videos[21:,0:],clases[21:])]


    # parametros de la red
    n_visible = 230300
    n_hidden  = 500
    batchSize = 10

    # creo la red
    red = RBM(n_visible=n_visible, n_hidden=n_hidden, ruta=rutaCompleta)
    #red.dibujarPesos(red.get_pesos... )
    #red.dibujarFiltros(nombreArchivo="filtritos.pdf")

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
    red.setParams({'epocas':10})


    T = temporizador()
    inicio = T.tic()

    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0:1], gibbsSteps=1)[0]
    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0], gibbsSteps=1)
    #MNIST.plot_one_digit((train_img/255.0).astype(numpy.float32)[0])
    #MNIST.plot_one_digit(salida)


    red.entrenamiento(data=datos[0][0],
                      miniBatchSize=batchSize,
                      pcd=False,
                      gibbsSteps=1,
                      validationData=datos[1][0],
                      filtros=True)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    # guardo los estadisticos
    #red.dibujarEstadisticos(show=True, save='estadisticos.png')
    #red.dibujarEstadisticos(show=True, save=rutaCompleta+'estadisticos.png')

    red.sampleo(data=datos[0][0],
                labels=datos[0][1])

    print('Guardando el modelo en ...', rutaCompleta)
    inicio = T.tic()
    red.guardar(nombreArchivo="rbm_mnist.zip")
    final = T.toc()
    print("Tiempo total para guardar: {}".format(T.transcurrido(inicio, final)))

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
