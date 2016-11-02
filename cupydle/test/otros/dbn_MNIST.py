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

"""
Pruebas sobre MNIST con una DBN

optirun python3 cupydle/test/dbn_MNIST.py "mnist_minmax.npz"
"""

def test():
    assert len(sys.argv) == 2, "cantidad incorrecta de parametros"

    parametros = {}
    parametros['general']         = ['mnist']
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
    parametros['porcentaje']      = [0.8]
    parametros['toleranciaError'] = [0.0]
    parametros['pcd']             = [True]
    parametros['directorio']      = ['dbn_mnist']
    parametros['dataset']         = [sys.argv[1]]
    parametros['capas']           = [[784, 500, 100, 10]]

    Grid = ParameterGrid(parametros)
    parametros = Grid[0]
    nombreArchivo = 'resultados_dbn_MNIST'
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

    entrenamiento        = datos['entrenamiento'].astype(np.float32)
    entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
    del datos # libera memoria

    datosDBN = []; datosDBN.append((entrenamiento, entrenamiento_clases))
    del entrenamiento, entrenamiento_clases

    ## se entrena
    tiempo_entrenar = d.entrenar(data=datosDBN)
    del datosDBN

    ###########################################################################
    ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
    ###########################################################################
    try:
        datos = np.load(rutaDatos + str(sys.argv[1]))
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + str(sys.argv[1])

    entrenamiento        = datos['entrenamiento'].astype(np.float32)
    entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
    validacion           = datos['validacion'].astype(np.float32)
    validacion_clases    = datos['validacion_clases'].astype(np.int32)
    testeo               = datos['testeo'].astype(np.float32)
    testeo_clases        = datos['testeo_clases'].astype(np.int32)
    del datos # libera memoria

    datosMLP = []; datosMLP.append((entrenamiento, entrenamiento_clases))
    datosMLP.append((validacion, validacion_clases)) ; datosMLP.append((testeo, testeo_clases))
    del entrenamiento, entrenamiento_clases, validacion, validacion_clases, testeo, testeo_clases

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
