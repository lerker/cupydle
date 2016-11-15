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

#---
# MNIST RESUMIDO
optirun python3 cupydle/test/dbn_prueba.py --dataset "mnist_minmax.npz" --directorio "test_DBN_mnist" --capa 784 500 10

#---
# MNIST COMPLETO
python3 cupydle/test/dbn_prueba.py \
--dataset "mnist_minmax.npz" \
--directorio "test_DBN_mnist" \
--capa 784 100 10 \
--epocasTRN 10 \
--epocasFIT 5 \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.1 \
--lrFIT 0.1 \
--gibbs 1 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria" \
--noEntrenar

#---
# KML RESUMIDO
optirun python3 cupydle/test/dbn_prueba.py --dataset "all_av_features_clases_shuffled_minmax.npz" --directorio "test_DBN_kml" --capa 230300 1000 6

#---
# KML PCA COMPLETO
python3 cupydle/test/dbn_prueba.py \
--dataset "all_av_features_clases_shuffled_minmax.npz" \
--directorio "test_DBN_kml" \
--capa 230300 1000 50 6 \
--epocasTRN 10 \
--epocasFIT 5 \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.1 \
--lrFIT 0.1 \
--gibbs 1 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria" \
--noEntrenar


#---
# KML PCA RESUMIDO
python3 cupydle/test/dbn_prueba.py --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" --directorio "test_DBN_kml" --capa 85 50 6

#---
# KML PCA COMPLETO
python3 cupydle/test/dbn_prueba.py \
--dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" \
--directorio "test_DBN_kml" \
--capa 85 50 6 \
--epocasTRN 11 \
--epocasFIT 4 \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.1 \
--lrFIT 0.1 \
--gibbs 1 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria" \
--noEntrenar

"""

#def test(general, directorio, dataset, capas, archivoResultados):
def test(archivoResultados, noEntrenar, **parametrosd):
    """
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
    """
    parametros=parametrosd
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
        datos = np.load(rutaDatos + parametros['dataset'])
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + parametros['dataset']

    KML = False
    if 'videos' in datos.keys():
        KML = True

    if KML:
        videos = datos['videos']
        clases = datos['clases'] - 1 # las clases estan desde 1..6, deben ser desde 0..5
        del datos # libera memoria

        # divido todo el conjunto con 120 ejemplos para el test
        # separo con un 86% aprox
        X_train, _, y_train, _ = train_test_split(videos, clases, test_size=120, random_state=np.random.randint(2**30))
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
    if not noEntrenar:
        tiempo_entrenar = d.entrenar(data=datosDBN)
    del datosDBN

    ###########################################################################
    ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
    ###########################################################################
    try:
        datos = np.load(rutaDatos + parametros['dataset'])
    except:
        assert False, "El dataset no existe en la ruta: " + rutaDatos + parametros['dataset']

    KML = False
    if 'videos' in datos.keys():
        KML = True

    if KML:
        videos = datos['videos']
        clases = datos['clases'] - 1 # las clases estan desde 1..6, deben ser desde 0..5
        del datos # libera memoria

        # divido todo el conjunto con 120 ejemplos para el test
        # separo con un 86% aprox
        X_train, X_test, y_train, y_test = train_test_split(videos, clases, test_size=120, random_state=np.random.randint(2**30))
        del videos, clases

        # me quedaron 600 ejemplos, lo divido de nuevo pero me quedo con 100 ejemplos para validacion
        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=100, random_state=np.random.randint(2**30))
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
    return costoTRN, costoVAL, costoTST, costoTST_final



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prueba de una DBN sobre MNIST/KML')
    parser.add_argument('--directorio', type=str,   dest="directorio", default='test_DBN', required=False, help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--dataset',    type=str,   dest="dataset",    default=None,       required=True,  help="Archivo donde esta el dataset, .npz")
    parser.add_argument('--capa',       type=int,   dest="capa",       default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")

    parser.add_argument('--noEntrenar', action="store_true",dest="noEntrenar",default=False, required=False, help="Si esta presente, no ejecuta el entrenamiento de la DBN, solo ajusta los pesos")
    parser.add_argument('--epocasTRN',  type=int,   dest="epocasTRN",  default=10,         required=False, nargs='+', help="Epocas de entrenamiento (no supervisado) para cada capa, si se especifica una sola se aplica para todas por igual")
    parser.add_argument('--epocasFIT',  type=int,   dest="epocasFIT",  default=10,         required=False, help="Epocas de para el ajuste (supervisado)")
    parser.add_argument('--batchsize',  type=int,   dest="batchsize",  default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('--porcentaje', type=float, dest="porcentaje", default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('--lrTRN',      type=float, dest="lrTRN",      default=0.1,        required=False, nargs='+', help="Tasas de aprendizaje para cada capa de entrenamiento, si se especifica una sola se aplica por igual a todas")
    parser.add_argument('--lrFIT',      type=float, dest="lrFIT",      default=0.01,       required=False, help="Tasa de aprendizaje para el ajuste de la red")
    parser.add_argument('--gibbs',      type=int,   dest="gibbs",      default=1,          required=False, nargs='+', help="Cantidad de pasos de Gibbs para la Divergencia Contrastiva, si es unitario se aplica para todos")
    parser.add_argument('--nombre',     type=str,   dest="nombre",     default='dbn',      required=False, help="Nombre del modelo")
    parser.add_argument('--pcd',        action="store_true",dest="pcd",default=False,      required=False, help="Habilita el algoritmo de entrenamiento Divergencia Contrastiva Persistente")
    parser.add_argument('--reguL1',     type=float, dest="reguL1",     default=0.0,        required=False, help="Parametro regularizador L1 para el costo del ajuste")
    parser.add_argument('--reguL2',     type=float, dest="reguL2",     default=0.0,        required=False, help="Parametro regularizador L2 para el costo del ajuste")
    parser.add_argument('--tolErr',     type=float, dest="tolErr",     default=0.0,        required=False, help="Criterio de parada temprana, tolerancia al error")
    parser.add_argument('--momentoTRN', type=float, dest="momentoTRN", default=0.0,        required=False, nargs='+', help="Tasa de momento para la etapa de entrenamiento, si es unico se aplica igual a cada capa")
    parser.add_argument('--momentoFIT', type=float, dest="momentoFIT", default=0.0,        required=False, help="Tasa de momento para la etapa de ajuste")
    parser.add_argument('--tipo',       type=str,   dest="tipo",       default='binaria',  required=False, help="Tipo de RBM (binaria, gaussiana)")

    argumentos = parser.parse_args()
    directorio = argumentos.directorio
    dataset    = argumentos.dataset
    epocasTRN  = argumentos.epocasTRN
    epocasFIT  = argumentos.epocasFIT
    batchsize  = argumentos.batchsize
    porcentaje = argumentos.porcentaje
    lrTRN      = argumentos.lrTRN
    lrFIT      = argumentos.lrFIT
    gibbs      = argumentos.gibbs
    nombre     = argumentos.nombre
    pcd        = argumentos.pcd
    reguL1     = argumentos.reguL1
    reguL2     = argumentos.reguL2
    tolErr     = argumentos.tolErr
    momentoTRN = argumentos.momentoTRN
    momentoFIT = argumentos.momentoFIT
    tipo       = argumentos.tipo
    noEntrenar = argumentos.noEntrenar

    capas = argumentos.capa #?

    #momentoTRN   = np.asarray([momentoTRN]) if isinstance(momentoTRN, float) else np.asarray(momentoTRN)
    #gibbs   = np.asarray([gibbs]) if isinstance(gibbs, int) else np.asarray(gibbs)
    archivoResultados="resultadosDBN"
    general = "kml"
    # es mnist o kml? segun el dataset
    if dataset.find("mnist") != -1:
        general = "mnist"

    # chequeos
    assert dataset.find('.npz') != -1, "El conjunto de datos debe ser del tipo '.npz'"
    #assert len(epocasTRN) >= len(capas) or len(epocasTRN) == 1, "Epocas de entrenamiento y cantidad de capas no coinciden (unidad aplica a todas)"
    #assert len(lrTRN) >= len(capas) or len(lrTRN) == 1, "Tasa de aprendizaje no coincide con la cantidad de capas (unidad aplica a todas)"
    #assert len(momentoTRN) >= len(capas) or len(momentoTRN) == 1, "Tasa de momento entrenamiento no coincide con la cantidad de capas (unidad aplica a todas)"
    #assert len(gibbs) >= len(capas) or len(gibbs) == 1, "Pasos de Gibbs no coinciden con la cantidad de capas (unidad aplica a todas)"
    assert tipo == 'binaria' or tipo == 'gaussiana', "Tipo de RBM no implementada"
    assert porcentaje <= 1.0

    """
    # ajustes
    epocasTRN = epocasTRN * len(capas) if len(epocasTRN) == 1 else epocasTRN
    tasaAprenTRN = np.resize(tasaAprenTRN, (len(capas),)) if len(tasaAprenTRN) == 1 else tasaAprenTRN
    momentoTRN = np.resize(momentoTRN, (len(capas),)) if len(momentoTRN) == 1 else momentoTRN
    pasosGibbs = np.resize(pasosGibbs, (len(capas),)) if len(pasosGibbs) == 1 else pasosGibbs
    """

    parametros = {}
    parametros['general']         = [general]
    parametros['nombre']          = [nombre]
    parametros['tipo']            = [tipo]
    parametros['capas']           = [capas]
    parametros['epocasTRN']       = [epocasTRN]
    parametros['epocasFIT']       = [epocasFIT]
    parametros['tambatch']        = [batchsize]
    parametros['tasaAprenTRN']    = [lrTRN]
    parametros['tasaAprenFIT']    = [lrFIT]
    parametros['regularizadorL1'] = [reguL1]
    parametros['regularizadorL2'] = [reguL2]
    parametros['momentoTRN']      = [momentoTRN]
    parametros['momentoFIT']      = [momentoFIT]
    parametros['pasosGibbs']      = [gibbs]
    parametros['porcentaje']      = [porcentaje]
    parametros['toleranciaError'] = [tolErr]
    parametros['pcd']             = [pcd]
    parametros['directorio']      = [directorio]
    parametros['dataset']         = [dataset]

    test(archivoResultados=archivoResultados, noEntrenar=noEntrenar, **parametros)


