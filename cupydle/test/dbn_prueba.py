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

import sys, os, argparse, numpy as np

from cupydle.test.dbn_base import dbn_base_test, _guardar
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.validacion_cruzada import train_test_split
from cupydle.dnn.utils import temporizador

"""
Pruebas sobre MNIST/KML con una DBN

# MNIST
optirun python3 cupydle/test/dbn_prueba.py --dataset "mnist_minmax.npz" --directorio "test_DBN_mnist" --capa 784 500 10

# KML
optirun python3 cupydle/test/dbn_prueba.py --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" --directorio "test_DBN_kml" --capa 85 50 6
"""

def test(general, directorio, dataset, capas, archivoResultados):

    parametros = {}
    parametros['general']         = [general]
    parametros['nombre']          = ['dbn']
    parametros['tipo']            = ['binaria']
    parametros['capas']           = []
    parametros['epocasTRN']       = [[50]]
    parametros['epocasFIT']       = [100]
    parametros['tambatch']        = [10]
    parametros['tasaAprenTRN']    = [0.01]
    parametros['tasaAprenFIT']    = [0.1]
    parametros['regularizadorL1'] = [0.0]
    parametros['regularizadorL2'] = [0.0]
    parametros['momentoTRN']      = [0.0]
    parametros['momentoFIT']      = [0.0]
    parametros['pasosGibbs']      = [5]
    parametros['porcentaje']      = [0.08]
    parametros['toleranciaError'] = [0.0]
    parametros['pcd']             = [True]
    parametros['directorio']      = [directorio]
    parametros['dataset']         = [dataset]
    parametros['capas']           = capas

    Grid = ParameterGrid(parametros)
    parametros = Grid[0]
    print("GUARDANDO LOS RESULTADOS EN EL ARCHIVO {}\n\n".format(archivoResultados))

    ##
    ##
    tiempo_entrenar = 0; tiempo_ajustar = 0; tiempo_total = 0
    T = temporizador() ; inicio = T.tic()
    directorioActual = os.getcwd() # directorio actual de ejecucion
    rutaDatos = str(directorioActual) + '/cupydle/data/DB_' + str(parametros['general']) +'/' # donde se almacenan la base de datos

    print("****************************************************")
    print("PARAMETROS:")
    for k in sorted(parametros.keys()):
        print(str("{: >20} : {: <20}").format(k, str(parametros[k])))

    # creo la dbn
    d = dbn_base_test(**parametros)

    ###########################################################################
    ##   P R E P A R A N D O   L O S   D A T O S   E N T R E N A M I E N T O
    ###########################################################################

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos keys, 'entrenamiento' y 'entrenamiento_clases'
    try:
        datos = np.load(rutaDatos + dataset)
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + dataset

    KML = False
    if 'videos' in datos.keys():
        KML = True

    if KML:
        videos = datos['videos']
        clases = datos['clases'] - 1 # las clases estan desde 1..6, deben ser desde 0..5
        del datos # libera memoria

        # divido todo el conjunto con 120 ejemplos para el test
        # separo con un 86% aprox
        X_train, _, y_train, _ = train_test_split(videos, clases, test_size=120, random_state=42)
        del videos, clases

        datosDBN = []; datosDBN.append((X_train, y_train))
        del X_train, y_train
    else:

        entrenamiento = datos['entrenamiento']
        entrenamiento_clases = datos['entrenamiento_clases'] #las clases creo que arrancan bien
        del datos # libera memoria

        entrenamiento = entrenamiento.astype(np.float32)
        entrenamiento_clases = entrenamiento_clases.astype(np.int32)
        datosDBN = []; datosDBN.append((entrenamiento, entrenamiento_clases))
        del entrenamiento, entrenamiento_clases

    ## se entrena, puede negarse con la variable de entorno "ENTRENAR"
    tiempo_entrenar = d.entrenar(data=datosDBN)
    del datosDBN

    ###########################################################################
    ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
    ###########################################################################
    try:
        datos = np.load(rutaDatos + dataset)
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + dataset

    KML = False
    if 'videos' in datos.keys():
        KML = True

    if KML:
        videos = datos['videos']
        clases = datos['clases'] - 1 # las clases estan desde 1..6, deben ser desde 0..5
        del datos # libera memoria

        # divido todo el conjunto con 120 ejemplos para el test
        # separo con un 86% aprox
        X_train, X_test, y_train, y_test = train_test_split(videos, clases, test_size=120, random_state=42)
        del videos, clases

        # me quedaron 600 ejemplos, lo divido de nuevo pero me quedo con 100 ejemplos para validacion
        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=100, random_state=42)
        del X_train, y_train

        datosMLP = []
        datosMLP.append((X_train_sub,y_train_sub)); datosMLP.append((X_valid,y_valid)); datosMLP.append((X_test,y_test))
        del X_train_sub, X_valid, y_train_sub, y_valid, X_test, y_test
    else:
        entrenamiento = np.asarray(datos['entrenamiento'], dtype=np.float32)
        validacion    = np.asarray(datos['validacion'], dtype=np.float32)
        testeo        = np.asarray(datos['testeo'], dtype=np.float32)
        entrenamiento_clases = np.asarray(datos['entrenamiento_clases'], dtype=np.int32)
        validacion_clases    = np.asarray(datos['validacion_clases'], dtype=np.int32)
        testeo_clases        = np.asarray(datos['testeo_clases'], dtype=np.int32)
        del datos

        datosMLP = []
        datosMLP.append((entrenamiento,entrenamiento_clases)); datosMLP.append((validacion,validacion_clases))
        datosMLP.append((testeo,testeo_clases))
        del entrenamiento,entrenamiento_clases,validacion,validacion_clases,testeo,testeo_clases

    ## se ajustan
    costoTRN, costoVAL, costoTST, costoTST_final, tiempo_ajustar = d.ajustar(datos=datosMLP)
    del datosMLP
    #

    #almaceno los resultados generales
    _guardar(nombreArchivo=archivoResultados, valor={'parametros':parametros, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final })
    print("\n\n")

    final = T.toc()
    tiempo_total = T.transcurrido(inicio, final)

    print("TIEMPOS [HH:MM:SS.ss]:")
    print("->  Entrenamiento: {}".format(T.transcurrido(0,tiempo_entrenar)))
    print("->  Ajuste:        {}".format(T.transcurrido(0,tiempo_ajustar)))
    print("->  Total:         {}".format(T.transcurrido(inicio, final)))
    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre MNIST/KML')
    parser.add_argument('--directorio', type=str,  dest="directorio", default='test_DBN', required=None,  help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',    type=str,  dest="dataset",    default=None,       required=True,  help="Archivo donde esta el dataset, .npz")
    parser.add_argument('--capa',       type=int,  dest="capa",       default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    argumentos = parser.parse_args()
    directorio = argumentos.directorio
    dataset    = argumentos.dataset
    capas = []
    if argumentos.capa is not None:
        capa = np.asarray(argumentos.capa)
        capas.append(capa)
    archivoResultados="resultadosDBN"

    general = "kml"
    # es mnist o kml? segun el dataset
    if dataset.find("mnist") != -1:
        general = "mnist"

    test(general, directorio, dataset, capas, archivoResultados)

