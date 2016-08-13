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
Implementacion de una Red de Creencia Profunda en GP-GPU/CPU (Theano)
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
from cupydle.dnn.dbn_gpu2 import dbn
from cupydle.dnn.dbn_gpu2 import rbmParams
from cupydle.dnn.funciones import sigmoideaTheano
from cupydle.dnn.mlp import MLP
## Data
from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk
## Utils
from cupydle.dnn.utils import temporizador


if __name__ == "__main__":

    directorioActual = os.getcwd()                               # directorio actual de ejecucion
    rutaTest    = directorioActual + '/cupydle/test/mnist/'      # sobre el de ejecucion la ruta a los tests
    rutaDatos    = directorioActual + '/cupydle/data/DB_mnist/'   # donde se almacenan la base de datos
    carpetaTest  = 'test_DBN/'                                  # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)


    subprocess.call(rutaTest + 'get_data.sh', shell=True)   # chequeo si necesito descargar los datos


    setName = "mnist"
    MNIST.prepare(rutaDatos, nombre=setName, compresion='bzip2')


    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre MNIST.')
    parser.add_argument('-r', '--rbm', action="store_true", dest="rbm", help="ejecutar las rbm", default=False)
    parser.add_argument('-m', '--mlp', action="store_true", dest="mlp", help="mlp simple", default=False)
    parser.add_argument('-d', '--dbnf', action="store_true", dest="dbnf", help="dbn fit", default=False)
    parser.add_argument('-e', '--modelo', action="store", dest="modelo", help="nombre del binario donde se guarda/abre el modelo", default="capa1.pgz")
    args = parser.parse_args()

    rbm = args.rbm
    mlp = args.mlp
    dbnf = args.dbnf
    modelName = args.modelo
    #modelName = 'capa1.pgz'
    verbose = True

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

    # [unidades_de_entrada, primer_capa_oculta, segunda_capa_oculta, salida]
    unidadesCapas = [784, 500, 100, 10]

    if mlp :
        print("S E C C I O N        M L P")

        batch_size=10

        clasificador = MLP( clasificacion=True,
                            rng=None,
                            ruta=rutaCompleta)

        clasificador.setParametroEntrenamiento({'tasaAprendizaje':0.01})
        clasificador.setParametroEntrenamiento({'regularizadorL1':0.00})
        clasificador.setParametroEntrenamiento({'regularizadorL2':0.0001})
        clasificador.setParametroEntrenamiento({'momento':0.0})
        clasificador.setParametroEntrenamiento({'epocas':1000})
        clasificador.setParametroEntrenamiento({'activationfuntion':sigmoideaTheano()})

        clasificador.agregarCapa(unidadesEntrada=unidadesCapas[0],
                                 unidadesSalida=unidadesCapas[1],
                                 clasificacion=False,
                                 activacion=sigmoideaTheano(),
                                 pesos=None,
                                 biases=None)

        clasificador.agregarCapa(#unidadesEntrada=500,
                                 unidadesSalida=unidadesCapas[2],
                                 clasificacion=False,
                                 activacion=sigmoideaTheano(),
                                 pesos=None,
                                 biases=None)

        clasificador.agregarCapa(#unidadesEntrada=100,
                                 unidadesSalida=unidadesCapas[3],
                                 clasificacion=True,
                                 activacion=sigmoideaTheano(),
                                 pesos=None,
                                 biases=None)

        T = temporizador()
        inicio = T.tic()

        # se almacenan los pesos para propositos de comparacion con la dbn
        numpy.save(rutaCompleta + "pesos1",clasificador.capas[0].W.get_value())
        numpy.save(rutaCompleta + "pesos2",clasificador.capas[1].W.get_value())
        numpy.save(rutaCompleta + "pesos3",clasificador.capas[2].W.get_value())

        clasificador.train( trainSet=datos[0],
                            validSet=datos[1],
                            testSet=datos[2],
                            batch_size=batch_size)

        final = T.toc()
        print("Tiempo total para entrenamiento MLP: {}".format(T.transcurrido(inicio, final)))

    if rbm :
        print("S E C C I O N        R B M")
        pasosGibbs=15
        numEpoch=100
        batchSize=10

        miDBN = dbn(name=None, ruta=rutaCompleta)


        # se cargan los pesos del mlp para comenzar desde ahi, y luego comparar con la dbn
        pesos1 = numpy.load(rutaCompleta + "pesos1.npy")
        pesos2 = numpy.load(rutaCompleta + "pesos2.npy")
        pesos3 = numpy.load(rutaCompleta + "pesos3.npy")

        # agrego una capa..
        miDBN.addLayer(n_visible=unidadesCapas[0],
                       n_hidden=unidadesCapas[1],
                       numEpoch=numEpoch,
                       batchSize=batchSize,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=pesos1)
        # otra capa mas
        miDBN.addLayer(#n_visible=500, # coincide con las ocultas de las anteriores
                       n_hidden=unidadesCapas[2],
                       numEpoch=numEpoch,
                       batchSize=batchSize,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=pesos2)

        # clasificacion
        miDBN.addLayer(#n_visible=100, # coincide con las ocultas de las anteriores
                       n_hidden=unidadesCapas[3],
                       numEpoch=numEpoch,
                       batchSize=batchSize,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=pesos3)

        T = temporizador()
        inicio = T.tic()

        #entrena la red
        miDBN.train(dataTrn=datos[0][0], # imagenes de entrenamiento
                    dataVal=datos[1][0]) # imagenes de validacion

        final = T.toc()
        print("Tiempo total para pre-entrenamiento DBN-(RBM): {}".format(T.transcurrido(inicio, final)))

        miDBN.save(rutaCompleta + "dbnMNIST", compression='zip')

    if dbnf:
        print("S E C C I O N        D B N")

        miDBN = dbn.load(filename=rutaCompleta + "dbnMNIST", compression='zip')
        print(miDBN)

        miDBN.fit(datos=datos,
                  listaPesos=None,
                  fnActivacion=sigmoideaTheano(),
                  semillaRandom=None)

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
