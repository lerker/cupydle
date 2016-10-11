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
GRID SEARCH
# ejemplo con dos capas del pca y directoria KML
# anda hasta con 5 capas

optirun python3 cupydle/test/dbn_KML_gridSearch.py --directorio KML --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" --capa1 85 60 6 --capa2 85 50 6

"""

# dependecias internar
import os, argparse, shelve, sys, argparse, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.dbn import DBN
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.validacion_cruzada import train_test_split

from cupydle.test.dbn_basica import DBN_basica, _guardar

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre KML GRID SEARCH')
    parser.add_argument('--directorio', type=str,  dest="directorio", default='test_DBN', required=None,  help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',    type=str,  dest="dataset",    default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('--capa1',      type=int,  dest="capa1",      default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa2',      type=int,  dest="capa2",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa3',      type=int,  dest="capa3",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa4',      type=int,  dest="capa4",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa5',      type=int,  dest="capa5",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    argumentos = parser.parse_args()
    directorio = argumentos.directorio
    dataset    = argumentos.dataset
    capas = []
    if argumentos.capa1 is not None:
        capa1 = np.asarray(argumentos.capa1)
        capas.append(capa1)
    if argumentos.capa2 is not None:
        capa2 = np.asarray(argumentos.capa2)
        capas.append(capa2)
    if argumentos.capa3 is not None:
        capa3 = np.asarray(argumentos.capa3)
        capas.append(capa3)
    if argumentos.capa4 is not None:
        capa4 = np.asarray(argumentos.capa4)
        capas.append(capa4)
    if argumentos.capa5 is not None:
        capa5 = np.asarray(argumentos.capa5)
        capas.append(capa5)

    #param_grid = {'a': [1, 2], 'b': [True, False]}
    # parametros de base
    parametros = {  'directorio':       [directorio],
                    'dataset':          [None],
                    'capas':            capas,
                    'epocasTRN':        [[100],[150],[200]],    # debe ser listas de listas
                    'epocasFIT':        [1000,1500,2000],
                    'tambatch':         [10,100],
                    'porcentaje':       [0.8],
                    'tasaAprenTRN':     [0.01, 0.05, 0.1],
                    'tasaAprenFIT':     [0.9,0.1],
                    'pasosGibbs':       [1],
                    'nombre':           ['dbn'],
                    'pcd':              [True, False],
                    'regularizadorL1':  [0.0],
                    'regularizadorL2':  [0.0],
                    'momentoTRN':       [0.0],
                    'momentoFIT':       [0.0, 0.1],
                    'tipo':             ['binaria', 'gaussiana'],
                    'toleranciaError':  [0.1]
                }

    #assert len(sys.argv) < 3, "cantidad incorrecta de parametros, la base de datos y una lista (opcional)"
    #parametros['dataset'] = ["all_videos_features_clases_shuffled_PCA85_minmax.npz"]
    parametros['dataset'] = [dataset]

    parametros['capas'] = capas

    Grid = ParameterGrid(parametros)
    #print(list(Grid))
    #print(len(Grid))
    cantidad_combinaciones = len(Grid)
    cantidad_a_ejecutar = cantidad_combinaciones // 2

    nombreArchivo = 'resultados_dbnFACE_gridSearch'
    print("GUARDANDO LOS RESULTADOS EN EL ARCHIVO {} QUE CONTIENE {} ITERACIONES\n\n".format(nombreArchivo, cantidad_a_ejecutar))

    T = temporizador()
    inicio = T.tic()

    for x in range(cantidad_a_ejecutar):
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
        costoTRN, costoVAL, costoTST, costoTST_final = DBN_basica(**params)
        print("****************************************************")
        print("\n\n")

        _guardar(nombreArchivo=nombreArchivo, diccionario={str(x): {'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})

    final = T.toc()
    print("\n\nGRID SEARCH  FINALIZADO\n\n")
    print("Tiempo total requerido: {}".format(T.transcurrido(inicio, final)))


