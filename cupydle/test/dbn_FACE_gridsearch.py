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

optirun python3 cupydle/test/dbn_FACE.py --directorio "test_DBN" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 50 6 --lepocaTRN 10 --lepocaFIT 100 -lrTRN 0.01 --unidadVis binaria --tolErr 0.08

"""

# dependecias internar
import os, argparse, shelve, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.dbn import DBN
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.validacion_cruzada import train_test_split

def ejecutar(**kwargs):

    # parametros pasados por consola
    directorio      = kwargs['directorio']
    dataset         = kwargs['dataset']
    capas           = kwargs['capas']
    epocasTRN       = kwargs['epocasTRN']
    epocasFIT       = kwargs['epocasFIT']
    tambatch        = kwargs['tambatch']
    porcentaje      = kwargs['porcentaje']
    tasaAprenTRN    = kwargs['tasaAprenTRN']
    tasaAprenFIT    = kwargs['tasaAprenFIT']
    pasosGibbs      = kwargs['pasosGibbs']
    nombre          = kwargs['nombre']
    pcd             = kwargs['pcd']
    regularizadorL1 = kwargs['regularizadorL1']
    regularizadorL2 = kwargs['regularizadorL2']
    momentoTRN      = kwargs['momentoTRN']
    momentoFIT      = kwargs['momentoFIT']
    unidadVis       = kwargs['unidadVis']
    toleranciaError = kwargs['toleranciaError']

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

    cantidad = int(clases.shape[0] * porcentaje)
    # la cantidad de ejemplos debe ser X partes enteras del minibatch, para que el algoritmo tome de a cachos y procese
    # es por ello que acomodo la cantidad hasta que quepa
    while (cantidad % tambatch):
        cantidad += 1
        assert cantidad != clases.shape[0], "Porcentaje trn/test muy alto, disminuir"
    cantidad_train = cantidad
    cantidad = int(cantidad_train * porcentaje)
    # la cantidad de ejemplos debe ser X partes enteras del minibatch, para que el algoritmo tome de a cachos y procese
    # es por ello que acomodo la cantidad hasta que quepa
    while (cantidad % tambatch):
        cantidad += 1
        assert cantidad != clases.shape[0], "Porcentaje trn/test muy alto, disminuir"
    cantidad_train2 = cantidad


    datosDBN = []
    datosMLP = []

    # entrena con todos los datos, test nunca los toca obvio
    datosTRN_DBN = (videos[:cantidad_train,:], clases[:cantidad_train])

    # entrena el MLP con una parte para train y otra validacion
    datosTRN = (videos[:cantidad_train2,:], clases[:cantidad_train2])
    datosVAL = (videos[cantidad_train2:cantidad_train,:],clases[cantidad_train2:cantidad_train])
    datosTST = (videos[cantidad_train:,:],clases[cantidad_train:])

    datosDBN.append(datosTRN_DBN);
    datosMLP.append(datosTRN); datosMLP.append(datosVAL); datosMLP.append(datosTST)

    print("                    Clases                     :", "[c1 c2 c3 c4 c5 c6]")
    print("                                               :", "-------------------")
    print("Cantidad de clases en el conjunto EntrenamieDBN:", np.bincount(datosTRN_DBN[1]))
    print("Cantidad de clases en el conjunto Entrenamiento:", np.bincount(datosTRN[1]))
    print("Cantidad de clases en el conjunto Validacion: \t", np.bincount(datosVAL[1]))
    print("Cantidad de clases en el conjunto Test: \t", np.bincount(datosTST[1]))
    print("Entrenado la DBN con {} ejemplos".format(len(datosTRN_DBN[0])))
    del datosTRN_DBN, datosTRN, datosVAL, datosTST

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
                       epocas=epocasTRN[idx],
                       tamMiniBatch=tambatch,
                       lr_pesos=tasaAprenTRN[idx],
                       pasosGibbs=pasosGibbs[idx],
                       w=None,
                       momento=momentoTRN[idx],
                       unidadesVisibles=unidadVis,
                       unidadesOcultas='binaria')

    #entrena la red
    miDBN.entrenar(dataTrn=datosDBN[0][0], # imagenes de entrenamiento
                   dataVal=None, # imagenes de validacion
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

    parametros={'tasaAprendizaje':  tasaAprenFIT,
                'regularizadorL1':  regularizadorL1,
                'regularizadorL2':  regularizadorL2,
                'momento':          momentoFIT}
    miDBN.setParametros(parametros)
    miDBN.setParametros({'epocas':epocasFIT})
    miDBN.setParametros({'toleranciaError':toleranciaError})

    costoTRN, costoVAL, costoTST, costoTST_final = miDBN.ajuste( datos=datosMLP,
                                                                 listaPesos=None,
                                                                 fnActivacion="sigmoidea",
                                                                 semillaRandom=None,
                                                                 tambatch=tambatch)

    miDBN.guardarObjeto(nombreArchivo=nombre)

    return costoTRN, costoVAL, costoTST, costoTST_final


def _guardar(nombreArchivo, diccionario=None):
        """
        Almacena todos los datos en un archivo pickle que contiene un diccionario
        lo cual lo hace mas sencillo procesar luego
        """
        archivo = nombreArchivo + '.cupydle'

        with shelve.open(archivo, flag='c', writeback=False, protocol=2) as shelf:
            for key in diccionario.keys():
                shelf[key] = diccionario[key]
            shelf.close()
        return 0

if __name__ == '__main__':

    #param_grid = {'a': [1, 2], 'b': [True, False]}
    # parametros de base
    parametros = {  'directorio':       ['dbn_dir'],
                    'dataset':          [None],
                    'capas':            [],
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
                    'unidadVis':        ['binaria', 'gaussiana'],
                    'toleranciaError':  [0.1]
                }

    parametros['dataset'] = ["all_videos_features_clases_shuffled_PCA85_minmax.npz"]
    parametros['capas'] = [[85, 50, 6], [85, 30, 6]]

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
        costoTRN, costoVAL, costoTST, costoTST_final = ejecutar(**params)
        print("****************************************************")
        print("\n\n")

        _guardar(nombreArchivo=nombreArchivo, diccionario={str(x): {'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})

    final = T.toc()
    print("\n\nGRID SEARCH  FINALIZADO\n\n")
    print("Tiempo total requerido: {}".format(T.transcurrido(inicio, final)))


