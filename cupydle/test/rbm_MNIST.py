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
Implementacion de una Maquina de Boltmann Restringida en GP-GPU/CPU (Theano)
Aplicado a la base de datos de 'benchmark' MNIST

http://yann.lecun.com/exdb/mnist/

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

    directorioActual= os.getcwd()                                   # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/mnist/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_mnist/'  # donde se almacenan la base de datos
    carpetaTest     = 'test_RBM/'                                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)

    if not os.path.exists(rutaDatos):
        print("Creando la base de datos en:", rutaDatos)
        os.makedirs(rutaDatos)

    # chequeo si necesito descargar los datos
    subprocess.call(rutaTest + 'get_data.sh', shell=True)

    setName = "mnist"
    MNIST.prepare(rutaDatos, nombre=setName, compresion='bzip2')

    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre MNIST.')
    parser.add_argument('-g', '--guardar', action="store_true", dest="guardar", help="desea guardar (correr desde cero)", default=False)
    parser.add_argument('-m', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    guardar = args.guardar
    modelName = args.modelo
    #modelName = 'capa1.pgz'

    # se leen de disco la base de datos
    mn = open4disk(filename=rutaDatos + setName, compression='bzip2')
    #mn.info

    # obtengo todos los subconjuntos
    #train_img,  train_labels= mn.get_training()
    #test_img,   test_labels = mn.get_testing()
    #val_img,    val_labels  = mn.get_validation()

    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...
    datos = [mn.get_training(), mn.get_testing(), mn.get_validation()]
    datos = [( ((x/255.0).astype(numpy.float32)), y) for x, y in datos]

    # parametros de la red
    n_visible = 784
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
    red.setParams({'epocas':100})


    T = temporizador()
    inicio = T.tic()

    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0:1], gibbsSteps=1)[0]
    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0], gibbsSteps=1)
    #MNIST.plot_one_digit((train_img/255.0).astype(numpy.float32)[0])
    #MNIST.plot_one_digit(salida)


    red.entrenamiento(data=datos[0][0],
                      validationData=datos[1][0],
                      tamMiniBatch=batchSize,
                      #tamMacroBatch=datos[1][0].shape[0]//2,
                      tamMacroBatch=None,
                      pcd=False,
                      gibbsSteps=1,
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


"""
lerker@nelo-linux: cupydle [memoria !]$  optirun python3 cupydle/test/rbm_MNIST.py
Using gpu device 0: GeForce GT 420M (CNMeM is disabled, cuDNN not available)
Entrenando una RBM, con [784] unidades visibles y [500] unidades ocultas
Cantidad de ejemplos para el entrenamiento no supervisado:  50000
Entrenando con Divergencia Contrastiva, 1 pasos de Gibss.
Unidades de visibles: Unidad Binaria Unidades Ocultas: Unidad Binaria
Epoca   1 de   5, error<TrnSet>:     inf, MSE<ejemplo> :     inf, EnergiaLibre<ejemplo>:     inf
Epoca   2 de   5, error<TrnSet>:-90.23674, MSE<ejemplo> : 130.21144, EnergiaLibre<ejemplo>:-3.74406
Epoca   3 de   5, error<TrnSet>:-82.94151, MSE<ejemplo> : 108.59375, EnergiaLibre<ejemplo>: 1.02042
Epoca   4 de   5, error<TrnSet>:-81.69224, MSE<ejemplo> : 105.26365, EnergiaLibre<ejemplo>: 1.78799
Epoca   5 de   5, error<TrnSet>:-80.99998, MSE<ejemplo> : 103.43192, EnergiaLibre<ejemplo>: 2.49729

Tiempo total para entrenamiento: 00:03:19.35
labels:  [5 5 9 1 0 8 1 6 2 5 3 7 4 0 5 9 6 1 7 3]
 ... plotting sample 0
 ... plotting sample 1
 ... plotting sample 2
 ... plotting sample 3
 ... plotting sample 4
 ... plotting sample 5
 ... plotting sample 6
 ... plotting sample 7
 ... plotting sample 8
 ... plotting sample 9
Guardando el modelo en ... /run/media/lerker/Documentos/Proyecto/Codigo/cupydle/cupydle/test/mnist/test_RBM/
Tiempo total para guardar: 00:00:00.43
ICE default IO error handler doing an exit(), pid = 2172, errno = 32
"""
