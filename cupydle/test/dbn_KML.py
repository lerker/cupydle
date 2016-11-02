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

import sys, os, numpy as np

from cupydle.test.dbn_base import dbn_base_test, _guardar
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.utils import temporizador
from cupydle.dnn.validacion_cruzada import train_test_split

"""
Pruebas sobre KML con una DBN

optirun python3 cupydle/test/dbn_KML.py "all_videos_features_clases_shuffled_PCA85_minmax.npz"
"""

def test():
    assert len(sys.argv) == 2, "cantidad incorrecta de parametros"

    parametros = {}
    parametros['general']         = ['kml']
    parametros['nombre']          = ['dbn']
    parametros['tipo']            = ['binaria']
    parametros['capas']           = []
    parametros['epocasTRN']       = [[5]]
    parametros['epocasFIT']       = [10]
    parametros['tambatch']        = [10]
    parametros['tasaAprenTRN']    = [0.01]
    parametros['tasaAprenFIT']    = [0.1]
    parametros['regularizadorL1'] = [0.0]
    parametros['regularizadorL2'] = [0.0]
    parametros['momentoTRN']      = [0.0]
    parametros['momentoFIT']      = [0.0]
    parametros['pasosGibbs']      = [1]
    parametros['porcentaje']      = [0.8]
    parametros['toleranciaError'] = [0.0]
    parametros['pcd']             = [True]
    parametros['directorio']      = ['dbn_kml']
    parametros['dataset']         = [sys.argv[1]]
    parametros['capas']           = [[85, 50, 15, 6]]

    Grid = ParameterGrid(parametros)
    parametros = Grid[0]
    nombreArchivo = 'resultados_dbn_KML'
    print("GUARDANDO LOS RESULTADOS EN EL ARCHIVO {}\n\n".format(nombreArchivo))

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
        datos = np.load(rutaDatos + str(sys.argv[1]))
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + str(sys.argv[1])

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

    ## se entrena, puede negarse con la variable de entorno "ENTRENAR"
    tiempo_entrenar = d.entrenar(data=datosDBN)
    del datosDBN

    ###########################################################################
    ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
    ###########################################################################
    try:
        datos = np.load(rutaDatos + str(sys.argv[1]))
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + str(sys.argv[1])

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

    ## se ajustan
    costoTRN, costoVAL, costoTST, costoTST_final, tiempo_ajustar = d.ajustar(datos=datosMLP)
    del datosMLP
    #

    #almaceno los resultados generales
    _guardar(nombreArchivo=nombreArchivo, valor={'parametros':parametros, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final })
    print("\n\n")

    final = T.toc()
    tiempo_total = T.transcurrido(inicio, final)

    print("TIEMPOS [HH:MM:SS.ss]:")
    print("->  Entrenamiento: {}".format(T.transcurrido(0,tiempo_entrenar)))
    print("->  Ajuste:        {}".format(T.transcurrido(0,tiempo_ajustar)))
    print("->  Total:         {}".format(T.transcurrido(inicio, final)))
    return 0

if __name__ == '__main__':
    test()
