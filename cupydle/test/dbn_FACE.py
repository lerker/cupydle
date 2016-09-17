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
import numpy
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
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-e1', '--epocasDBN', type=int,   dest="epocasDBN",      default=10,         required=False, help="Cantidad de epocas para el entrenamiento de cada capa RBM")
    parser.add_argument('-e2', '--epocasMLP', type=int,   dest="epocasMLP",      default=10,         required=False, help="Cantidad de epocas para el entrenamiento del mlp ajuste fino")
    parser.add_argument('-p', '--porcentaje', type=float, dest="porcentaje",     default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-l', '--capas',      type=int,   dest="capas",          default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('-lrDBN',             type=float, dest="tasaAprenDBN",   default=0.01,       required=False, help="Tasa de aprendizaje (General) para todos las capas de la RBM")
    parser.add_argument('-lrMLP',             type=float, dest="tasaAprenMLP",   default=0.01,       required=False, help="Tasa de aprendizaje para el ajuste de los pesos en el gradiente estocastico del MLP ajuste fino")

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
    capas = numpy.asarray(capas)

    # configuraciones con respecto a los directorios
    directorioActual= os.getcwd()                                  # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest     = directorio_ejecucion + '/'                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    if not os.path.exists(rutaCompleta):        # si no existe la crea
        print('Creando la carpeta para el test en: ',rutaCompleta)
        os.makedirs(rutaCompleta)

    ###########################################################################
    ##
    ##        M A N I P U L A C I O N    DE   L O S    D A T O S
    ##
    ###########################################################################

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos archivos, 'videos' y 'clases'
    b = numpy.load(rutaDatos + dataset)
    videos = b['videos']
    clases = b['clases']
    # las clases estan desde 1..6, deben ser desde 0..5
    clases -= 1
    del b #libera memoria

    # modo hardcore activatedddd
    datos = []
    datosTRN = (videos[0:550,:], clases[0:550])
    datosVAL = (videos[550:670,:],clases[550:670])
    datosTST = (videos[670:,:],clases[670:])
    datos.append(datosTRN); datos.append(datosVAL); datos.append(datosTST)


    ###########################################################################
    ##
    ##          P R E - E N T R E N A M I E N T O        R B M s
    ##
    ###########################################################################

    pasosGibbs=1

    miDBN = DBN(name=None, ruta=rutaCompleta)

    # agrego una capa..
    for idx, _ in enumerate(capas[:-1]): # es -2 porque no debo tener en cuenta la primera ni la ultima
        miDBN.addLayer(n_visible=capas[idx],
                       n_hidden=capas[idx+1],
                       numEpoch=epocasDBN,
                       tamMiniBatch=tambatch,
                       epsilonw=0.1,
                       pasosGibbs=pasosGibbs,
                       w=None,
                       unidadesVisibles=UnidadBinaria(),
                       unidadesOcultas=UnidadBinaria())

    #entrena la red
    miDBN.entrenar(dataTrn=datos[0][0], # imagenes de entrenamiento
                   dataVal=datos[1][0], # imagenes de validacion
                   pcd=False,
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

    parametros={'tasaAprendizaje':tasaAprenMLP,
                'regularizadorL1':0.00,
                'regularizadorL2':0.0001,
                'momento':0.0,
                'activationfuntion':sigmoideaTheano()}
    miDBN.setParametrosAjuste(parametros)

    miDBN.setParametrosAjuste({'epocas':epocasMLP})
    #miDBN.setParametrosAjuste({'toleranciaError':0.08})

    miDBN.ajuste(datos=datos,
                 listaPesos=None,
                 fnActivacion=sigmoideaTheano(),
                 semillaRandom=None)

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
