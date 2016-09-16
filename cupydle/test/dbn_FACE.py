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
Implementacion de una Red de Creencia Profunda en GP-GPU/CPU (Theano) FACE
"""
import numpy
import time
import os
import sys
import subprocess
import argparse

# Dependencias Externas
## Core
from cupydle.dnn.dbn import DBN
from cupydle.dnn.dbn import rbmParams
from cupydle.dnn.funciones import sigmoideaTheano
from cupydle.dnn.mlp import MLP
## Data
from cupydle.test.mnist.mnist import MNIST
from cupydle.test.mnist.mnist import open4disk
from cupydle.test.mnist.mnist import save2disk
## Utils
from cupydle.dnn.utils import temporizador
from cupydle.dnn.unidades import UnidadBinaria


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre FACE')
    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_DBN', required=None,  help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-e1', '--epocasDBN', type=int,   dest="epocasDBN",      default=10,         required=False, help="Cantidad de epocas para el entrenamiento de cada capa RBM")
    parser.add_argument('-e2', '--epocasMLP', type=int,   dest="epocasMLP",      default=10,         required=False, help="Cantidad de epocas para el entrenamiento del mlp ajuste fino")
    parser.add_argument('-p', '--porcentaje', type=float, dest="porcentaje",     default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-l', '--capas',      type=int,   dest="capas",          default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('-lrDBN',             type=float, dest="tasaAprenDBN",   default=0.01,       required=False, help="Tasa de aprendizaje (General) para todos las capas de la RBM")
    parser.add_argument('-lrMLP',             type=float, dest="tasaAprenMLP",   default=0.01,       required=False, help="Tasa de aprendizaje para el ajuste de los pesos en el gradiente estocastico del MLP ajuste fino")

    parser.add_argument('-m', '--mlp',        dest="mlp", action="store_true",   default=False,      required=False, help="Ejecuta un MLP inicial para observar las mejoras")
    parser.add_argument('-r', '--rbm',        dest="rbm", action="store_true",   default=False,      required=False, help="Ejecuta un RBM para buscar la mejora de los pesos anteriores")
    parser.add_argument('-d', '--dbn',        dest="dbn", action="store_true",   default=False,      required=False, help="Ejecuta un DBN para chequear si se mejoraron los pesos gracias al preentrenamiento")
    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio_ejecucion  = argumentos.directorio
    dataset               = argumentos.dataset
    tambatch              = argumentos.tambatch
    epocasDBN             = argumentos.epocasDBN
    epocasMLP             = argumentos.epocasMLP
    porcentaje            = argumentos.porcentaje
    capas                 = argumentos.capas
    tasaAprenDBN          = argumentos.tasaAprenDBN
    tasaAprenMLP          = argumentos.tasaAprenMLP
    seccionMLP            = argumentos.mlp
    seccionRBM            = argumentos.rbm
    seccionDBN            = argumentos.dbn

    # configuraciones con respecto a los directorios
    directorioActual= os.getcwd()                                  # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest     = directorio_ejecucion + '/'                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos archivos, 'videos' y 'clases'
    b = numpy.load(rutaDatos + dataset)
    videos = b['videos']
    clases = b['clases']
    # las clases estan desde 1..6, deben ser desde 0..5
    clases -= 1
    del b #libera memoria

    if mlp :
        print("S E C C I O N        M L P")
        print("\nSe entrena un multilayer perceptron para chequear la mejora")

        clasificador = MLP( clasificacion=True,
                            rng=None,
                            ruta=rutaCompleta)

        clasificador.setParametroEntrenamiento({'tasaAprendizaje':tasaAprenMLP})
        clasificador.setParametroEntrenamiento({'regularizadorL1':0.00})
        clasificador.setParametroEntrenamiento({'regularizadorL2':0.0001})
        clasificador.setParametroEntrenamiento({'momento':0.0})
        clasificador.setParametroEntrenamiento({'epocas':epocasMLP})
        clasificador.setParametroEntrenamiento({'activationfuntion':sigmoideaTheano()})
        clasificador.setParametroEntrenamiento({'toleranciaError':0.00})

        # agrego tantas capas desee de regresion
        # la primera es la que indica la entrada
        # las intermedias son de regresion
        # la ultima es de clasificaicon
        for idx, _ in enumerate(capas[:-2]): # es -2 porque no debo tener en cuenta la primera ni la ultima
            clasificador.agregarCapa(unidadesEntrada=capas[idx], unidadesSalida=capas[idx+1], clasificacion=False, activacion=sigmoideaTheano(), pesos=None, biases=None)
        clasificador.agregarCapa(unidadesSalida=capas[-1], clasificacion=True, pesos=None, biases=None)

        T = temporizador()
        inicio = T.tic()

        # se almacenan los pesos para propositos de comparacion con la dbn
        for idx, _ in enumerate(capas):
            numpy.save(rutaCompleta + "pesos_" + str(idx), clasificador.capas[idx].getW())

        # se entrena la red
        errorTRN, errorVAL, errorTST = clasificador.entrenar(trainSet=datos[0],
                                                             validSet=datos[1],
                                                             testSet=datos[2],
                                                             batch_size=tambatch)

        final = T.toc()
        print("Tiempo total para entrenamiento MLP: {}".format(T.transcurrido(inicio, final)))

    if rbm :
        print("S E C C I O N        R B M")
        pasosGibbs=1
        numEpocas=5
        batchSize=10

        miDBN = DBN(name=None, ruta=rutaCompleta)


        # se cargan los pesos del mlp para comenzar desde ahi, y luego comparar con la dbn
        pesos1 = numpy.load(rutaCompleta + "pesos1.npy")
        pesos2 = numpy.load(rutaCompleta + "pesos2.npy")
        pesos3 = numpy.load(rutaCompleta + "pesos3.npy")

        # agrego una capa..
        miDBN.addLayer(n_visible=unidadesCapas[0],
                       n_hidden=unidadesCapas[1],
                       numEpoch=numEpocas,
                       tamMiniBatch=batchSize,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=pesos1,
                       unidadesVisibles=UnidadBinaria(),
                       unidadesOcultas=UnidadBinaria())
        # otra capa mas
        miDBN.addLayer(#n_visible=500, # coincide con las ocultas de las anteriores
                       n_hidden=unidadesCapas[2],
                       numEpoch=numEpocas,
                       tamMiniBatch=batchSize,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=pesos2,
                       unidadesVisibles=UnidadBinaria(),
                       unidadesOcultas=UnidadBinaria())

        # clasificacion
        miDBN.addLayer(#n_visible=100, # coincide con las ocultas de las anteriores
                       n_hidden=unidadesCapas[3],
                       numEpoch=numEpocas,
                       tamMiniBatch=batchSize,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=pesos3,
                       unidadesVisibles=UnidadBinaria(),
                       unidadesOcultas=UnidadBinaria())

        T = temporizador()
        inicio = T.tic()

        #entrena la red
        miDBN.preEntrenamiento(dataTrn=datos[0][0], # imagenes de entrenamiento
                               dataVal=datos[1][0], # imagenes de validacion
                               pcd=False,
                               guardarPesosIniciales=True,
                               filtros=True)

        final = T.toc()
        print("Tiempo total para pre-entrenamiento DBN-(RBM): {}".format(T.transcurrido(inicio, final)))

        miDBN.save(rutaCompleta + "dbnMNIST", compression='zip')

    if dbnf:
        print("S E C C I O N        D B N")

        miDBN = DBN.load(filename=rutaCompleta + "dbnMNIST", compression='zip')
        print(miDBN)

        parametros={'tasaAprendizaje':0.01,
                    'regularizadorL1':0.00,
                    'regularizadorL2':0.0001,
                    'momento':0.0,
                    'activationfuntion':sigmoideaTheano()}
        miDBN.setParametrosAjuste(parametros)

        miDBN.setParametrosAjuste({'epocas':10})
        #miDBN.setParametrosAjuste({'toleranciaError':0.08})

        miDBN.ajuste(datos=datos,
                     listaPesos=None,
                     fnActivacion=sigmoideaTheano(),
                     semillaRandom=None)

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
