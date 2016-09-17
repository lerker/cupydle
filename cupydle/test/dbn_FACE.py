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

Esta es una prueba simple de una DBN completa sobre la base de datos RML
Ejecuta el entrenamiento no supervisado y luego ajuste fino de los parametros
por medio de un entrenamiento supervisado.

"""
import numpy as np
import os
import argparse

# Dependencias Externas
## Core
from cupydle.dnn.dbn import DBN
# TODO implementar dentro de mlp y elegir a traves de un string
from cupydle.dnn.funciones import sigmoideaTheano
from cupydle.dnn.mlp import MLP

# TODO implementar dentro de dbn
from cupydle.dnn.unidades import UnidadBinaria


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre FACE')
    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_DBN', required=None,  help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-l', '--capas',      type=int,   dest="capas",          default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('-let', '--lepocaTRN',type=int,   dest="epocasTRN",      default=10,         required=False, nargs='+', help="Epocas de entrenamiento (no supervisado) para cada capa, si se especifica una sola se aplica para todas por igual")
    parser.add_argument('-lea', '--lepocaFIT',type=int,   dest="epocasFIT",      default=10,         required=False, help="Epocas de para el ajuste (supervisado)")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-p', '--porcentaje', type=float, dest="porcentaje",     default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-lrTRN',             type=float, dest="tasaAprenTRN",   default=0.01,       required=False, nargs='+', help="Tasas de aprendizaje para cada capa de entrenamiento, si se especifica una sola se aplica por igual a todas")
    parser.add_argument('-lrFIT',             type=float, dest="tasaAprenFIT",   default=0.01,       required=False, help="Tasa de aprendizaje para el ajuste de la red")
    parser.add_argument('-g', '--gibbs',      type=int,   dest="pasosGibbs",     default=1,          required=False, nargs='+', help="Cantidad de pasos de Gibbs para la Divergencia Contrastiva, si es unitario se aplica para todos")
    parser.add_argument('--nombre',           type=str,   dest="nombre",         default=None,       required=False, help="Nombre del modelo")
    parser.add_argument('--pcd',              action="store_true", dest="pcd",   default=False,      required=False, help="Habilita el algoritmo de entrenamiento Divergencia Contrastiva Persistente")
    parser.add_argument('--reguL1',           type=float, dest="regularizadorL1",default=0.0,        required=False, help="Parametro regularizador L1 para el costo del ajuste")
    parser.add_argument('--reguL2',           type=float, dest="regularizadorL2",default=0.0,        required=False, help="Parametro regularizador L2 para el costo del ajuste")
    parser.add_argument('--momentoTRN',       type=float, dest="momentoTRN",     default=0.0,        required=False, nargs='+', help="Tasa de momento para la etapa de entrenamiento, si es unico se aplica igual a cada capa")
    parser.add_argument('--momentoFIT',       type=float, dest="momentoFIT",     default=0.0,        required=False, help="Tasa de momento para la etapa de ajuste")

    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio      = argumentos.directorio
    dataset         = argumentos.dataset
    capas           = argumentos.capas
    epocasTRN       = argumentos.epocasTRN
    epocasFIT       = argumentos.epocasFIT
    tambatch        = argumentos.tambatch
    porcentaje      = argumentos.porcentaje
    tasaAprenTRN    = argumentos.tasaAprenTRN
    tasaAprenFIT    = argumentos.tasaAprenFIT
    pasosGibbs      = argumentos.pasosGibbs
    nombre          = argumentos.nombre
    pcd             = argumentos.pcd
    regularizadorL1 = argumentos.regularizadorL1
    regularizadorL2 = argumentos.regularizadorL2
    momentoTRN      = argumentos.momentoTRN
    momentoFIT      = argumentos.momentoFIT

    capas        = np.asarray(capas)
    tasaAprenTRN = np.asarray([tasaAprenTRN]) if isinstance(tasaAprenTRN, float) else np.asarray(tasaAprenTRN)
    momentoTRN   = np.asarray([momentoTRN]) if isinstance(momentoTRN, float) else np.asarray(momentoTRN)
    pasosGibbs   = np.asarray([pasosGibbs]) if isinstance(pasosGibbs, int) else np.asarray(pasosGibbs)

    # chequeos
    assert dataset.find('.npz') != -1, "El conjunto de datos debe ser del tipo '.npz'"
    assert len(epocasTRN) >= len(capas) or len(epocasTRN) == 1, "Epocas de entrenamiento y cantidad de capas no coinciden (unidad aplica a todas)"
    assert len(tasaAprenTRN) >= len(capas) or len(tasaAprenTRN) == 1, "Tasa de aprendizaje no coincide con la cantidad de capas (unidad aplica a todas)"
    assert len(momentoTRN) >= len(capas) or len(momentoTRN) == 1, "Tasa de momento entrenamiento no coincide con la cantidad de capas (unidad aplica a todas)"
    assert len(pasosGibbs) >= len(capas) or len(pasosGibbs) == 1, "Pasos de Gibbs no coinciden con la cantidad de capas (unidad aplica a todas)"
    assert porcentaje <= 1.0

    # ajustes
    epocasTRN = epocasTRN * len(capas) if len(epocasTRN) == 1 else epocasTRN
    tasaAprenTRN = np.resize(tasaAprenTRN, (len(capas),)) if len(tasaAprenTRN) == 1 else tasaAprenTRN
    momentoTRN = np.resize(momentoTRN, (len(capas),)) if len(momentoTRN) == 1 else momentoTRN
    pasosGibbs = np.resize(pasosGibbs, (len(capas),)) if len(pasosGibbs) == 1 else pasosGibbs


    # configuraciones con respecto a los directorios
    directorioActual = os.getcwd()                                  # directorio actual de ejecucion
    rutaTest         = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos        = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest      = directorio + '/'                   # carpeta a crear para los tests
    rutaCompleta     = rutaTest + carpetaTest
    os.makedirs(rutaCompleta) if not os.path.exists(rutaCompleta) else None # crea la carpeta en tal caso que no exista

    ###########################################################################
    ##
    ##        M A N I P U L A C I O N    DE   L O S    D A T O S
    ##
    ###########################################################################

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos keys, 'videos' y 'clases'
    try:
        datos = np.load(rutaDatos + dataset)
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + dataset

    videos = datos['videos']
    clases = datos['clases'] - 1 # las clases estan desde 1..6, deben ser desde 0..5
    del datos # libera memoria

    # modo hardcore activatedddd
    # aplico el mismo porcentaje para dividir el conjunto de los datos en train y test
    # y el mismo porcentaje para train propio y validacion
    cantidad_train  = int(len(clases) * porcentaje)
    cantidad_train2 = int(cantidad_train * porcentaje)

    datos = []
    datosTRN = (videos[:cantidad_train2,:], clases[:cantidad_train2])
    datosVAL = (videos[cantidad_train2:cantidad_train,:],clases[cantidad_train2:cantidad_train])
    datosTST = (videos[cantidad_train:,:],clases[cantidad_train:])
    datos.append(datosTRN); datos.append(datosVAL); datos.append(datosTST)
    del datosTRN, datosVAL, datosTST

    ###########################################################################
    ##
    ##          P R E - E N T R E N A M I E N T O        R B M s
    ##
    ###########################################################################

    # se crea el modelo
    miDBN = DBN(name=nombre, ruta=rutaCompleta)

    # se agregan las capas
    for idx in range(len(capas[:-1])): # es -2 porque no debo tener en cuenta la primera ni la ultima
        miDBN.addLayer(n_visible=capas[idx],
                       n_hidden=capas[idx+1],
                       numEpoch=epocasTRN[idx],
                       tamMiniBatch=tambatch,
                       epsilonw=tasaAprenTRN[idx],
                       pasosGibbs=pasosGibbs[idx],
                       w=None,
                       momentum=momentoTRN[idx],
                       unidadesVisibles=UnidadBinaria(),
                       unidadesOcultas=UnidadBinaria())

    #entrena la red

    miDBN.entrenar(dataTrn=datos[0][0], # imagenes de entrenamiento
                   dataVal=datos[1][0], # imagenes de validacion
                   pcd=pcd,
                   guardarPesosIniciales=True,
                   filtros=True)

    #miDBN.save(rutaCompleta + "dbnMNIST", compression='zip')

    ###########################################################################
    ##
    ##                 A J U S T E     F I N O    ( M L P )
    ##
    ###########################################################################

    #miDBN = DBN.load(filename=rutaCompleta + "dbnMNIST", compression='zip')
    print(miDBN)

    parametros={'tasaAprendizaje':tasaAprenFIT,
                'regularizadorL1':regularizadorL1,
                'regularizadorL2':regularizadorL2,
                'momento':momentoFIT,
                'activationfuntion':sigmoideaTheano()}
    miDBN.setParametrosAjuste(parametros)

    miDBN.setParametrosAjuste({'epocas':epocasFIT})
    #miDBN.setParametrosAjuste({'toleranciaError':0.08})

    miDBN.ajuste(datos=datos,
                 listaPesos=None,
                 fnActivacion=sigmoideaTheano(),
                 semillaRandom=None,
                 tambatch=tambatch)

    miDBN.guardar(nombre)
else:
    assert False, "Esto no es un modulo, es un TEST!!!"
