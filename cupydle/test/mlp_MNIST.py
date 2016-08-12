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
from cupydle.dnn.activations import Sigmoid
## Data
from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk
## Utils
from cupydle.dnn.utils import temporizador

from cupydle.dnn.mlp import MLP

if __name__ == "__main__":

    directorioActual= os.getcwd()                                   # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/mnist/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_mnist/'  # donde se almacenan la base de datos
    carpetaTest     = 'test_MLP/'                                   # carpeta a crear para los tests
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

    batchSize = 10
    n_epochs = 1

    # [unidades_de_entrada, primer_capa_oculta, segunda_capa_oculta, salida]
    unidadesCapas = [784, 500, 100, 10]

    # creo la red
    #MNIST.plot_one_digit(train_img.get_value()[0])
    clasificador = MLP( clasificacion=True,
                        rng=None,
                        ruta=rutaCompleta)

    clasificador.setParametroEntrenamiento({'tasaAprendizaje':0.01})
    clasificador.setParametroEntrenamiento({'regularizadorL1':0.00})
    clasificador.setParametroEntrenamiento({'regularizadorL2':0.0001})
    clasificador.setParametroEntrenamiento({'momento':0.0})
    clasificador.setParametroEntrenamiento({'epocas':1000})
    clasificador.setParametroEntrenamiento({'activationfuntion':Sigmoid()})

    clasificador.agregarCapa(unidadesEntrada=784, unidadesSalida=500, clasificacion=False, activacion=Sigmoid(), pesos=None, biases=None)
    clasificador.agregarCapa(unidadesSalida=10, clasificacion=True, pesos=None, biases=None)

    T = temporizador()
    inicio = T.tic()



    clasificador.train( trainSet=datos[0],
                        validSet=datos[1],
                        testSet=datos[2],
                        batch_size=batchSize)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))


else:
    assert False, "Esto no es un modulo, es un TEST!!!"
