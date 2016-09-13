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

    directorioActual= os.getcwd()                                   # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest     = 'test_MLP/'                                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)

    if not os.path.exists(rutaDatos):
        print("Creando la base de datos en:", rutaDatos)
        os.makedirs(rutaDatos)

    setName = "face"

    parser = argparse.ArgumentParser(description='Prueba de un MLP sobre FACE.')
    parser.add_argument('-g', '--guardar', action="store_true", dest="guardar", help="desea guardar (correr desde cero)", default=False)
    parser.add_argument('-m', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    guardar = args.guardar
    modelName = args.modelo
    #modelName = 'capa1.pgz'

    b = numpy.load(rutaDatos + 'videos_clases_procesados_zscore_minmax.npz')
    videos = b['videos']
    clases = b['clases']
    del b

    # obtengo todos los subconjuntos
    #train_img,  train_labels= mn.get_training()
    #test_img,   test_labels = mn.get_testing()
    #val_img,    val_labels  = mn.get_validation()

    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...
    #datos = [(videos[0:20,0:], clases[0:20]), (videos[21:,0:],clases[21:])]
    datos = [(videos, clases), (videos[100:200,0:],clases[100:200]), (videos[200:300,0:],clases[200:300:])]

    batchSize = 10
    n_epochs = 1

    # [unidades_de_entrada, primer_capa_oculta, segunda_capa_oculta, salida]
    unidadesCapas = [230300, 500, 100, 10]

    # creo la red
    #MNIST.plot_one_digit(train_img.get_value()[0])
    clasificador = MLP( clasificacion=True,
                        rng=None,
                        ruta=rutaCompleta)

    clasificador.setParametroEntrenamiento({'tasaAprendizaje':0.01})
    clasificador.setParametroEntrenamiento({'regularizadorL1':0.00})
    clasificador.setParametroEntrenamiento({'regularizadorL2':0.0001})
    clasificador.setParametroEntrenamiento({'momento':0.0})
    clasificador.setParametroEntrenamiento({'activationfuntion':sigmoideaTheano()})
    clasificador.setParametroEntrenamiento({'epocas':10})
    clasificador.setParametroEntrenamiento({'toleranciaError':0.08})

    clasificador.agregarCapa(unidadesEntrada=784, unidadesSalida=500, clasificacion=False, activacion=sigmoideaTheano(), pesos=None, biases=None)
    clasificador.agregarCapa(unidadesSalida=10, clasificacion=True, pesos=None, biases=None)

    T = temporizador()
    inicio = T.tic()



    clasificador.train( trainSet=datos[0],
                        validSet=datos[1],
                        testSet=datos[2],
                        batch_size=batchSize)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    # guardando los parametros aprendidos
    clasificador.guardarParametros()

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
