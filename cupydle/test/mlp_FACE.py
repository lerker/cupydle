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
# dependencias python
import os, argparse, numpy as np

# dependecias externas
from cupydle.dnn.funciones import sigmoideaTheano
from cupydle.dnn.utils import temporizador
from cupydle.dnn.mlp import MLP

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de un MLP sobre KLM')
    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_MLP', required=None,  help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-l', '--capas',      type=int,   dest="capas",          default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('-let', '--lepocaTRN',type=int,   dest="epocasTRN",      default=10,         required=False, help="cantidad de epocas de entrenamiento")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-p', '--porcentaje', type=float, dest="porcentaje",     default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-lrTRN',             type=float, dest="tasaAprenTRN",   default=0.01,       required=False, help="Tasa de aprendizaje para la red")
    parser.add_argument('--nombre',           type=str,   dest="nombre",         default=None,       required=False, help="Nombre del modelo")
    parser.add_argument('--reguL1',           type=float, dest="regularizadorL1",default=0.0,        required=False, help="Parametro regularizador L1 para el costo del ajuste")
    parser.add_argument('--reguL2',           type=float, dest="regularizadorL2",default=0.0,        required=False, help="Parametro regularizador L2 para el costo del ajuste")
    parser.add_argument('--momentoTRN',       type=float, dest="momentoTRN",     default=0.0,        required=False, help="Tasa de momento para la etapa de entrenamiento")
    parser.add_argument('--tolError',         type=float, dest="tolError",       default=0.0,        required=False, help="Toleracia al error como criterio de parada temprana")

    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio      = argumentos.directorio
    dataset         = argumentos.dataset
    capas           = argumentos.capas
    epocasTRN       = argumentos.epocasTRN
    tambatch        = argumentos.tambatch
    porcentaje      = argumentos.porcentaje
    tasaAprenTRN    = argumentos.tasaAprenTRN
    nombre          = argumentos.nombre
    regularizadorL1 = argumentos.regularizadorL1
    regularizadorL2 = argumentos.regularizadorL2
    momentoTRN      = argumentos.momentoTRN
    tolError        = argumentos.tolError

    capas        = np.asarray(capas)
    tasaAprenTRN = np.float32(tasaAprenTRN)
    momentoTRN   = np.float(momentoTRN)
    tolError     = np.float(tolError)

    # chequeos
    assert dataset.find('.npz') != -1, "El conjunto de datos debe ser del tipo '.npz'"
    assert porcentaje <= 1.0

    # configuraciones con respecto a los directorios
    directorioActual = os.getcwd()                                  # directorio actual de ejecucion
    rutaTest         = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos        = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest      = directorio + '/'                   # carpeta a crear para los tests
    rutaCompleta     = rutaTest + carpetaTest
    os.makedirs(rutaCompleta) if not os.path.exists(rutaCompleta) else None # crea la carpeta en tal caso que no exista


    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos archivos, 'videos' y 'clases'
    b = np.load(rutaDatos + dataset)
    videos = b['videos'];               clases = b['clases']
    # las clases estan desde 1..6, deben ser desde 0..5
    clases -= 1
    del b #libera memoria


    # hardcodedd... puede utilizarce otro criterio
    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...
    datos = []
    datosTRN = (videos[0:550,:], clases[0:550])
    datosVAL = (videos[550:670,:],clases[550:670])
    datosTST = (videos[670:,:],clases[670:])
    datos.append(datosTRN); datos.append(datosVAL); datos.append(datosTST)

    # creo la red
    clasificador = MLP(clasificacion=True,
                       rng=None,
                       ruta=rutaCompleta)

    # se agregan los parametros
    parametros = {'tasaAprendizaje':  tasaAprenTRN,
                  'regularizadorL1':  regularizadorL1,
                  'regularizadorL2':  regularizadorL2,
                  'momento':          momentoTRN,
                  'activationfuntion':sigmoideaTheano(),
                  'epocas':           epocasTRN,
                  'toleranciaError':  tolError}
    clasificador.setParametroEntrenamiento(parametros)

    # agrego tantas capas desee de regresion
    # la primera es la que indica la entrada
    # las intermedias son de regresion
    # la ultima es de clasificaicon
    for idx, _ in enumerate(capas[:-2]): # es -2 porque no debo tener en cuenta la primera ni la ultima
        print(idx, capas[idx], capas[idx+1])
        clasificador.agregarCapa(unidadesEntrada=capas[idx], unidadesSalida=capas[idx+1], clasificacion=False, activacion=sigmoideaTheano(), pesos=None, biases=None)
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
