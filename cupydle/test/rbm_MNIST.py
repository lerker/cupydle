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
from cupydle.dnn.activations import Sigmoid
from cupydle.dnn.unidades import UnidadBinaria
## Data
from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk
## Utils
from cupydle.dnn.utils import temporizador

from cupydle.dnn.rbm_gpu2 import RBM

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

    red.setParams({'epsilonw':0.1})
    red.setParams({'epsilonvb':0.1})
    red.setParams({'epsilonhb':0.1})
    red.setParams({'momentum':0.0})
    red.setParams({'weightcost':0.0})
    red.setParams({'maxepoch':3})
    red.setParams({'unidadesEntrada':UnidadBinaria()})
    red.setParams({'unidadesSalida':UnidadBinaria()})


    T = temporizador()
    inicio = T.tic()

    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0:1], gibbsSteps=1)[0]
    #salida = red.reconstruccion(vsample=(train_img/255.0).astype(numpy.float32)[0], gibbsSteps=1)
    #MNIST.plot_one_digit((train_img/255.0).astype(numpy.float32)[0])
    #MNIST.plot_one_digit(salida)


    red.train(  data=datos[0][0],
                miniBatchSize=batchSize,
                pcd=True,
                gibbsSteps=1,
                validationData=datos[1][0],
                filtros=True)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    # guardo los estadisticos
    #red.dibujarEstadisticos(show=True, save='estadisticos.png')
    #red.dibujarEstadisticos(show=True, save=rutaCompleta+'estadisticos.png')

    #red.sampleo(data=datos[0][0],
    #            labels=datos[0][1])

    print('Guardando el modelo en ...', rutaCompleta)
    inicio = T.tic()
    red.guardar(nombreArchivo="rbm_mnist.zip")
    final = T.toc()
    print("Tiempo total para guardar: {}".format(T.transcurrido(inicio, final)))

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
