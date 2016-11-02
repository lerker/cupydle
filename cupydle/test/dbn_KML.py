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

from cupydle.test.dbn_base import dbn_base_test, parametros, _guardar
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.utils import temporizador

def test():
    assert len(sys.argv) == 2, "cantidad incorrecta de parametros"
    parametros['dataset'] = [sys.argv[1]]
    parametros['capas'] = [[784, 100, 10]]
    #parametros['dataset'] = ["mnist_minmax.npz"]
    #parametros['capas'] = [[784, 100, 10], [85, 30, 6]]

    Grid = ParameterGrid(parametros)
    cantidad_combinaciones = len(Grid)
    cantidad_a_ejecutar = cantidad_combinaciones // 2

    nombreArchivo = 'resultados_dbnMNIST_gridSearch'
    print("GUARDANDO LOS RESULTADOS EN EL ARCHIVO {} QUE CONTIENE {} ITERACIONES\n\n".format(nombreArchivo, cantidad_a_ejecutar))

    T = temporizador()
    inicio = T.tic()

    for x in range(cantidad_a_ejecutar):
        T2 = temporizador()
        inicio2 = T2.tic()

        print("****************************************************")
        print("\n\n")
        print("Iteracion {} de {}".format(x, cantidad_a_ejecutar))
        print("PARAMETROS:")
        # modo random la eleccion de los parametros sobre el conjunto posible
        indice = np.random.randint(cantidad_combinaciones)
        # no tengo implementada el __setitem__ en ParameterGrid
        params = Grid[indice]
        params['directorio'] = 'dbn_grid_' + str(x)
        for k in sorted(params.keys()):
            print(str("{: >25} : {: <50}").format(k, str(params[k])))
        print("\n\n")
        d = dbn_base_test(**params)

        ##
        ###########################################################################
        ##
        ##   P R E P A R A N D O   L O S   D A T O S   E N T R E N A M I E N T O
        ##
        ###########################################################################

        # se cargan  los datos, debe ser un archivo comprimido, en el cual los
        # arreglos estan en dos keys, 'videos' y 'clases'
        directorioActual = os.getcwd() # directorio actual de ejecucion
        rutaDatos = str(directorioActual) + '/cupydle/data/DB_' + str(parametros['general'][0]) +'/' # donde se almacenan la base de datos
        try:
            datos = np.load(rutaDatos + str(sys.argv[1]))
        except:
            assert False, "El dataset no existe en la ruta: " + rutaDatos + str(sys.argv[1])

        entrenamiento        = datos['entrenamiento'].astype(np.float32)
        entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
        del datos # libera memoria

        datosDBN = []; datosDBN.append((entrenamiento, entrenamiento_clases))
        del entrenamiento, entrenamiento_clases
        ##
        d.entrenar(data=datosDBN)
        del datosDBN


        ##
        ###########################################################################
        ##
        ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
        ##
        ###########################################################################

        # se cargan  los datos, debe ser un archivo comprimido, en el cual los
        # arreglos estan en dos keys, 'videos' y 'clases'
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

        datosMLP = []
        datosMLP.append((entrenamiento, entrenamiento_clases))
        datosMLP.append((validacion, validacion_clases))
        datosMLP.append((testeo, testeo_clases))
        del entrenamiento, entrenamiento_clases, validacion, validacion_clases
        del testeo, testeo_clases
        ##
        costoTRN, costoVAL, costoTST, costoTST_final = d.ajustar(datos=datosMLP)
        del datosMLP
        #

        _guardar(nombreArchivo=nombreArchivo, valor={str(x): {'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})
        #_guardar(nombreArchivo=nombreArchivo, valor={str(x):{'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})
        #from cupydle.dnn.utils import cargarSHELVE
        #print(cargarSHELVE(nombreArchivo=nombreArchivo, clave=None))

        final2 = T2.toc()
        print("\n\nPaso de la grilla terminado, iteracion " + str(x+1) + "\n\n")
        print("Tiempo: {}".format(T2.transcurrido(inicio2, final2)))
        print("****************************************************")
        print("\n\n")

    final = T.toc()
    print("\n\nGRID SEARCH  FINALIZADO\n\n")
    print("Tiempo total requerido: {}".format(T.transcurrido(inicio, final)))


    return 0

if __name__ == '__main__':
    test()
