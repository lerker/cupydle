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
Estructura basica para la implementacion basica de una DBN sobre una base de datos KML

optirun python3 cupydle/test/dbn_basica.py "all_videos_features_clases_shuffled_PCA85_minmax.npz"

"""

# dependecias internar
import os, argparse, shelve, sys, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.dbn import DBN
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.validacion_cruzada import train_test_split

Parametros = {  'directorio':       'dbn_dir',
                'dataset':          'dbn_data',
                'capas':            [],
                'epocasTRN':        0,    # debe ser listas de listas
                'epocasFIT':        0,
                'tambatch':         0.0,
                'porcentaje':       0.0,
                'tasaAprenTRN':     0.0,
                'tasaAprenFIT':     0.0,
                'pasosGibbs':       0.0,
                'nombre':           'dbn',
                'pcd':              False,
                'regularizadorL1':  0.0,
                'regularizadorL2':  0.0,
                'momentoTRN':       0.0,
                'momentoFIT':       0.0,
                'tipo':             'binaria',
                'toleranciaError':  0.1
            }

def DBN_basica(**kwargs):

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
    tipo            = kwargs['tipo']
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
    ##   P R E P A R A N D O   L O S   D A T O S   E N T R E N A M I E N T O
    ##
    ###########################################################################

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos keys, 'videos' y 'clases'
    try:
        datos = np.load(rutaDatos + dataset)
    except:
        try:
            rutaTest         = directorioActual + '/cupydle/test/mnist/'     # sobre el de ejecucion la ruta a los tests
            rutaDatos        = directorioActual + '/cupydle/data/DB_mnist/'  # donde se almacenan la base de datos
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

        #validacion = datos['validacion']
        #validacion_clases = datos['validacion_clases'] #las clases creo que arrancan bien

        #testeo = datos['testeo']
        #testeo_clases = datos['testeo_clases'] #las clases creo que arrancan bien
        del datos # libera memoria

        entrenamiento = entrenamiento.astype(np.float32)
        entrenamiento_clases = entrenamiento_clases.astype(np.int32)
        datosDBN = []; datosDBN.append((entrenamiento, entrenamiento_clases))
        del entrenamiento, entrenamiento_clases

    ###########################################################################
    ##
    ##          P R E - E N T R E N A M I E N T O        R B M s
    ##
    ###########################################################################
    print("                    Clases                     :", "[c1 c2 c3 c4 c5 c6]")
    print("                                               :", "-------------------")
    print("Cantidad de clases en el conjunto EntrenamieDBN:", np.bincount(datosDBN[0][1]))
    print("Entrenado la DBN con {} ejemplos".format(len(datosDBN[0][1])))

    # se crea el modelo
    miDBN = DBN(nombre=nombre, ruta=rutaCompleta)

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
                       tipo=tipo)

    #entrena la red
    miDBN.entrenar(dataTrn=datosDBN[0][0], # imagenes de entrenamiento
                   dataVal=None, # imagenes de validacion
                   pcd=pcd,
                   guardarPesosIniciales=False,
                   filtros=False)

    #miDBN.save(rutaCompleta + "dbnMNIST", compression='zip')
    del datosDBN
    # FIN DEL ENTRENAMIENTO


    ###########################################################################
    ##
    ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
    ##
    ###########################################################################

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos keys, 'videos' y 'clases'
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

        entrenamiento = datos['entrenamiento']
        entrenamiento_clases = datos['entrenamiento_clases'] #las clases creo que arrancan bien

        validacion = datos['validacion']
        validacion_clases = datos['validacion_clases'] #las clases creo que arrancan bien

        testeo = datos['testeo']
        testeo_clases = datos['testeo_clases'] #las clases creo que arrancan bien
        del datos # libera memoria

        entrenamiento = entrenamiento.astype(np.float32)
        entrenamiento_clases = entrenamiento_clases.astype(np.int32)

        validacion = validacion.astype(np.float32)
        validacion_clases = validacion_clases.astype(np.int32)

        testeo = testeo.astype(np.float32)
        testeo_clases = testeo_clases.astype(np.float32)

        datosMLP = []
        datosMLP.append((entrenamiento,entrenamiento_clases)); datosMLP.append((validacion,validacion_clases)); datosMLP.append((testeo,testeo_clases))
        del entrenamiento,entrenamiento_clases,validacion,validacion_clases,testeo,testeo_clases

    ###########################################################################
    ##
    ##                 A J U S T E     F I N O    ( M L P )
    ##
    ###########################################################################
    print("                    Clases                     :", "[c1 c2 c3 c4 c5 c6]")
    print("                                               :", "-------------------")
    print("Cantidad de clases en el conjunto Entrenamiento:", np.bincount(datosMLP[0][1]))
    print("Cantidad de clases en el conjunto Validacion: \t", np.bincount(datosMLP[1][1]))
    print("Cantidad de clases en el conjunto Test: \t", np.bincount(datosMLP[2][1]))

    #miDBN = DBN.load(filename=rutaCompleta + "dbnMNIST", compression='zip')
    print(miDBN)

    parametros={'tasaAprendizaje':  tasaAprenFIT,
                'regularizadorL1':  regularizadorL1,
                'regularizadorL2':  regularizadorL2,
                'momento':          momentoFIT,
                'toleranciaError':  toleranciaError,
                'epocas':           epocasFIT}
    miDBN.setParametros(parametros)

    costoTRN, costoVAL, costoTST, costoTST_final = miDBN.ajuste( datos=datosMLP,
                                                                 listaPesos=None,
                                                                 fnActivacion="sigmoidea",
                                                                 semillaRandom=None,
                                                                 tambatch=tambatch)

    del datosMLP

    miDBN.guardarObjeto(nombreArchivo=nombre)

    # FIN DEL AJUSTE FINO

    return costoTRN, costoVAL, costoTST, costoTST_final


def _guardar(nombreArchivo, valor):
    from cupydle.dnn.utils import guardarHDF5
    import os.path

    nombreArchivo = nombreArchivo + '.cupydle'

    """
    if not os.path.isfile(nombreArchivo):

        datos={ 'parametros': [],
                'costoTRN': [],
                'costoVAL': [],
                'costoTST': [],
                'costoTST_final': []}
        guardarHDF5(nombreArchivo=nombreArchivo, valor=datos, nuevo=True)


    guardarHDF5(nombreArchivo=nombreArchivo, valor=valor)
    """
    with shelve.open(nombreArchivo, flag='c', writeback=False, protocol=2) as shelf:
            for key in valor.keys():
                shelf[key] = valor[key]
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
                    'tipo':             ['binaria', 'gaussiana'],
                    'toleranciaError':  [0.1]
                }

    assert len(sys.argv) == 2, "cantidad incorrecta de parametros"
    #parametros['dataset'] = ["all_videos_features_clases_shuffled_PCA85_minmax.npz"]
    parametros['dataset'] = [sys.argv[1]]
    parametros['capas'] = [[85, 50, 6], [85, 30, 6]]
    #parametros['capas'] = [[230346, 100, 50, 6], [230346, 1000, 500, 100, 6]]

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

        _guardar(nombreArchivo=nombreArchivo, valor={str(x): {'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})

    final = T.toc()
    print("\n\nGRID SEARCH  FINALIZADO\n\n")
    print("Tiempo total requerido: {}".format(T.transcurrido(inicio, final)))


