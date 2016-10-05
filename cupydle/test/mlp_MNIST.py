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
Script de pruebas sobre la base de datos MNIST con un modelo de regresion lineal

Simplemente carga los datos y ajusta la red con los parametros pasados


# version corta
optirun python3 cupydle/test/mlp_MNIST.py --directorio "test_MLP" --dataset "mnist_minmax.npz" -l 784 500 10 --lepocaTRN 10 -lrTRN 0.01 --tolError 0.08 --MLPrapido

# version larga
optirun python3 cupydle/test/mlp_MNIST.py --directorio "test_MLP" --dataset "mnist_minmax.npz" -l 784 500 10 --lepocaTRN 10 -b 100 -lrTRN 0.01 --tolError 0.01 --reguL1 0.0 --reguL2 0.0 --momentoTRN 0.0
"""

# dependencias internas
import os, argparse, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.mlp import MLP

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de un MLP sobre MNIST')
    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_MLP', required=False, help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--nombre',           type=str,   dest="nombre",         default='mlp',      required=False, help="Nombre del modelo")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-l', '--capas',      type=int,   dest="capas",          default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('-let', '--lepocaTRN',type=int,   dest="epocasTRN",      default=10,         required=False, help="cantidad de epocas de entrenamiento")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-lrTRN',             type=float, dest="tasaAprenTRN",   default=0.01,       required=False, help="Tasa de aprendizaje para la red")
    parser.add_argument('--reguL1',           type=float, dest="regularizadorL1",default=0.0,        required=False, help="Parametro regularizador L1 para el costo del ajuste")
    parser.add_argument('--reguL2',           type=float, dest="regularizadorL2",default=0.0,        required=False, help="Parametro regularizador L2 para el costo del ajuste")
    parser.add_argument('--momentoTRN',       type=float, dest="momentoTRN",     default=0.0,        required=False, help="Tasa de momento para la etapa de entrenamiento")
    parser.add_argument('--tolError',         type=float, dest="tolError",       default=0.0,        required=False, help="Toleracia al error como criterio de parada temprana")
    parser.add_argument('--MLPrapido',        action="store_true", dest="mlprapido", default=False,  required=False, help="Activa el entrenamiento rapido para el MLP, no testea todos los conjuntos de test, solo los mejores validaciones")

    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio      = argumentos.directorio
    dataset         = argumentos.dataset
    capas           = argumentos.capas
    epocasTRN       = argumentos.epocasTRN
    tambatch        = argumentos.tambatch
    tasaAprenTRN    = argumentos.tasaAprenTRN
    nombre          = argumentos.nombre
    regularizadorL1 = argumentos.regularizadorL1
    regularizadorL2 = argumentos.regularizadorL2
    momentoTRN      = argumentos.momentoTRN
    tolError        = argumentos.tolError
    mlprapido       = argumentos.mlprapido

    capas        = np.asarray(capas)
    tasaAprenTRN = np.float32(tasaAprenTRN)
    momentoTRN   = np.float(momentoTRN)
    tolError     = np.float(tolError)

    # chequeos
    assert dataset.find('.npz') != -1, "El conjunto de datos debe ser del tipo '.npz'"

    # configuraciones con respecto a los directorios
    directorioActual = os.getcwd()            # directorio actual de ejecucion
    rutaTest         = directorioActual + '/cupydle/test/mnist/' # ruta tests
    rutaDatos        = directorioActual + '/cupydle/data/DB_mnist/' #ruta DB
    carpetaTest      = directorio + '/'       # carpeta a crear para los tests
    rutaCompleta     = rutaTest + carpetaTest
    os.makedirs(rutaCompleta) if not os.path.exists(rutaCompleta) else None

    # carga los datos
    datos = np.load(rutaDatos + dataset)

    entrenamiento        = datos['entrenamiento'].astype(np.float32)
    entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
    validacion           = datos['validacion'].astype(np.float32)
    validacion_clases    = datos['validacion_clases'].astype(np.int32)
    testeo               = datos['testeo'].astype(np.float32)
    testeo_clases        = datos['testeo_clases'].astype(np.int32)
    del datos # libera memoria

    datosMLP = []
    datosMLP.append((entrenamiento, entrenamiento_clases))
    datosMLP.append((validacion, validacion_clases))
    datosMLP.append((testeo, testeo_clases))
    del entrenamiento, entrenamiento_clases, validacion, validacion_clases
    del testeo, testeo_clases

    # creo la red
    clasificador = MLP(ruta=rutaCompleta, nombre=nombre)

    # se agregan los parametros
    parametros = {'tasaAprendizaje':  tasaAprenTRN,
                  'regularizadorL1':  regularizadorL1,
                  'regularizadorL2':  regularizadorL2,
                  'momento':          momentoTRN,
                  'epocas':           epocasTRN,
                  'toleranciaError':  tolError}
    clasificador.setParametroEntrenamiento(parametros)

    for idx, _ in enumerate(capas[:-2]): # es -2 porque no debo tener en cuenta la primera ni la ultima
        clasificador.agregarCapa(unidadesEntrada=capas[idx], unidadesSalida=capas[idx+1], clasificacion=False, activacion='sigmoidea', pesos=None, biases=None)
    clasificador.agregarCapa(unidadesSalida=capas[-1], clasificacion=True, pesos=None, biases=None)

    T = temporizador()
    inicio = T.tic()

    ###########################################################################
    ##
    ##                 A J U S T E     F I N O    ( M L P )
    ##
    ###########################################################################
    print("                    Clases                     :", "[1    2    3    4    5    6    7    8    9    10]")
    print("                                               :", "-------------------------------")
    print("Cantidad de clases en el conjunto Entrenamiento:", np.bincount(datosMLP[0][1]))
    print("Cantidad de clases en el conjunto Validacion: \t", np.bincount(datosMLP[1][1]))
    print("Cantidad de clases en el conjunto Test: \t", np.bincount(datosMLP[2][1]))

    clasificador.entrenar(trainSet=datosMLP[0], validSet=datosMLP[1], testSet=datosMLP[2], batch_size=tambatch, rapido=mlprapido)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    clasificador.score(datos=datosMLP[2], guardar='Matriz de Confusion')

    # dibujar estadisticos
    clasificador.dibujarEstadisticos(mostrar=False, guardar=rutaCompleta+'estadisticosMLP')

    # guardando los parametros aprendidos
    clasificador.guardarObjeto(nombreArchivo=nombre)

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
