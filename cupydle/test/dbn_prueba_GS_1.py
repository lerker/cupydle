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

# anda hasta con 5 capas

optirun python3 cupydle/test/dbn_prueba_GS.py --directorio KML --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" --capa1 85 60 6 --capa2 85 50 6

optirun python3 cupydle/test/dbn_prueba_GS.py --directorio MNIST --dataset "mnist_minmax.npz" --capa1 784 500 10 --capa2 784 100 10
"""

# dependecias internar
import os, argparse, shelve, sys, argparse, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.test.dbn_base import _guardar
from cupydle.test.dbn_prueba import test

#
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre MNIST/RML GRID SEARCH')
    parser.add_argument('--directorio', type=str,  dest="directorio", default='test_DBN', required=None,  help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',    type=str,  dest="dataset",    default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('--capa1',      type=int,  dest="capa1",      default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa2',      type=int,  dest="capa2",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa3',      type=int,  dest="capa3",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa4',      type=int,  dest="capa4",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--capa5',      type=int,  dest="capa5",      default=None,       required=False, nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('--noEntrenar', action="store_true",dest="noEntrenar",default=False, required=False, help="Si esta presente, no ejecuta el entrenamiento de la DBN, solo ajusta los pesos")
    parser.add_argument('--cantidad',   type=int,  dest="cantidad",   default=100,       required=False,  help="Porcentaje de la cantidad total de iteraciones del grid search")


    argumentos = parser.parse_args()
    directorio = argumentos.directorio
    dataset    = argumentos.dataset
    noEntrenar = argumentos.noEntrenar
    cantidad   = argumentos.cantidad

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

    general = "kml"
    # es mnist o kml? segun el dataset
    if dataset.find("mnist") != -1:
        general = "mnist"

    # chequeos
    assert dataset.find('.npz') != -1, "El conjunto de datos debe ser del tipo '.npz'"

    parametros = {}
    parametros['general']         = [general]
    parametros['nombre']          = ['dbn']
    parametros['tipo']            = ['binaria', 'gaussiana']
    parametros['capas']           = capas
    parametros['epocasTRN']       = [[50]]
    parametros['epocasFIT']       = [500]
    parametros['tambatch']        = [10, 50, 100]
    parametros['tasaAprenTRN']    = [[0.01],[0.001],[0.0001],[0.05]]
    parametros['tasaAprenFIT']    = [0.1, 0.2, 0.01, 0.05]
    parametros['regularizadorL1'] = [0.0, 0.01, 0.1]
    parametros['regularizadorL2'] = [0.0, 0.001, 0.0001]
    parametros['momentoTRN']      = [[0.0], [0.1], [0.001]]
    parametros['momentoFIT']      = [0.0, 0.1, 0.2, 0.5]
    parametros['pasosGibbs']      = [[1], [5], [10]]
    parametros['porcentaje']      = [0.8]
    parametros['toleranciaError'] = [0.0]
    parametros['pcd']             = [True, False]
    parametros['directorio']      = [directorio]
    parametros['dataset']         = [dataset]
    parametros['dropout']         = [0.5]

    #parametros={'pasosGibbs': [[1]], 'pcd': [True], 'dataset': ['all_videos_features_clases_shuffled_PCA85_minmax.npz'], 'general': ['kml'], 'nombre': ['dbn'], 'regularizadorL1': [0.0], 'directorio': ['test_DBN_kml'], 'toleranciaError': [0.02], 'momentoFIT': [0.0], 'capas': [[85, 50, 6]], 'tipo': ['binaria'], 'epocasTRN': [[11]], 'tasaAprenFIT': [0.1], 'porcentaje': [0.8], 'momentoTRN': [[0.0]], 'epocasFIT': [4], 'tasaAprenTRN': [[0.1]], 'tambatch': [10], 'regularizadorL2': [0.0]}

    Grid = ParameterGrid(parametros)
    cantidad_combinaciones = len(Grid)
    cantidad_a_ejecutar = int(cantidad_combinaciones * cantidad / 100.0)

    nombreArchivo = 'resultadosGeneralesGS'
    print("GUARDANDO LOS RESULTADOS EN EL ARCHIVO {} QUE CONTIENE {} ITERACIONES\n\n".format(nombreArchivo, cantidad_a_ejecutar))

    T = temporizador()
    inicio = T.tic()

    for x in range(cantidad_a_ejecutar):
        print("Iteracion {} de {}".format(x, cantidad_a_ejecutar))
        # modo random la eleccion de los parametros sobre el conjunto posible
        indice = np.random.randint(cantidad_combinaciones)
        # no tengo implementada el __setitem__ en ParameterGrid
        params = Grid[indice]
        params['directorio'] = directorio + '/' + 'dbn_grid_' + str(x)
        for k in sorted(params.keys()):
            #print(str("{: >25} : {: <50}").format(k, str(params[k])))
            params[k] = [params[k]] # paso a lista todos los valores uno por uno
        archivoResultados1 = 'cupydle/test/' + params['general'][0] + '/' + params['directorio'][0] + '/' + nombreArchivo
        archivoResultados2 = 'cupydle/test/' + params['general'][0] + '/' + directorio + '/' + nombreArchivo

        costoTRN, costoVAL, costoTST, costoTST_final= test(archivoResultados=archivoResultados, noEntrenar=noEntrenar, **params)
        _guardar(nombreArchivo=archivoResultados2, valor={str(x): {'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})
        print("*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+")
        print("\n")

    final = T.toc()
    print("\n\nGRID SEARCH  FINALIZADO\n\n")
    print("Tiempo total requerido: {}".format(T.transcurrido(inicio, final)))


