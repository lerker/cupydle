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
optirun python3 cupydle/test/mlp_FACE_validacionCruzada.py --directorio "test_MLP_CV" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 50 6 --lepocaTRN 50 -lrTRN 0.01 --tolError 0.08 --directorioFold k_fold -c 6 --MLPrapido
"""
# dependencias internas
import os, argparse, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.mlp import MLP
from cupydle.dnn.validacion_cruzada import StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prueba de un MLP sobre KLM -- VALIDACION CRUZADA --')
    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_MLP_CV',required=False, help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--nombre',           type=str,   dest="nombre",         default='mlp',      required=False, help="Nombre del modelo")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-l', '--capas',      type=int,   dest="capas",          default=None,       required=True,  nargs='+', help="Capas de unidades [visibles, ocultas1.. ocultasn]")
    parser.add_argument('-let', '--lepocaTRN',type=int,   dest="epocasTRN",      default=10,         required=False, help="cantidad de epocas de entrenamiento")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-p', '--porcentaje', type=float, dest="porcentaje",     default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-lrTRN',             type=float, dest="tasaAprenTRN",   default=0.01,       required=False, help="Tasa de aprendizaje para la red")
    parser.add_argument('--reguL1',           type=float, dest="regularizadorL1",default=0.0,        required=False, help="Parametro regularizador L1 para el costo del ajuste")
    parser.add_argument('--reguL2',           type=float, dest="regularizadorL2",default=0.0,        required=False, help="Parametro regularizador L2 para el costo del ajuste")
    parser.add_argument('--momentoTRN',       type=float, dest="momentoTRN",     default=0.0,        required=False, help="Tasa de momento para la etapa de entrenamiento")
    parser.add_argument('--tolError',         type=float, dest="tolError",       default=0.0,        required=False, help="Toleracia al error como criterio de parada temprana")
    parser.add_argument('--MLPrapido',        action="store_true", dest="mlprapido", default=False,  required=False, help="Activa el entrenamiento rapido para el MLP, no testea todos los conjuntos de test, solo los mejores validaciones")
    parser.add_argument('--directorioFold',   type=str,   dest="directorioConjunto", default='k_fold',required=False, help="Nombre de la carpeta donde se almacenan individualmente cada corrida")
    parser.add_argument('-c', '--conjuntos',  type=int,   dest="conjuntos",      default=6,          required=False,     help="Cantidad de conjuntos para la validacion cruzada")

    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio      = argumentos.directorio
    dataset         = argumentos.dataset
    capas           = argumentos.capas
    epocasTRN       = argumentos.epocasTRN
    tambatch        = argumentos.tambatch
    porcentaje      = argumentos.porcentaje
    tasaAprenTRN    = argumentos.tasaAprenTRN
    nombre          = argumentos.nombre
    regularizadorL1 = argumentos.regularizadorL1
    regularizadorL2 = argumentos.regularizadorL2
    momentoTRN      = argumentos.momentoTRN
    tolError        = argumentos.tolError
    mlprapido       = argumentos.mlprapido
    conjuntos       = argumentos.conjuntos
    carpetaConjunto = argumentos.directorioConjunto

    # chequeos
    capas           = np.asarray(capas)
    tasaAprenTRN    = np.float32(tasaAprenTRN)
    momentoTRN      = np.float(momentoTRN)
    tolError        = np.float(tolError)
    assert porcentaje <= 1.0, "El porcenje no puede ser superior a 1"


    # configuraciones con respecto a los directorios
    directorioActual= os.getcwd()                                  # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/face/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_face/'  # donde se almacenan la base de datos
    carpetaTest     = directorio + '/'                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    os.makedirs(rutaCompleta) if not os.path.exists(rutaCompleta) else None

    # se cargan  los datos, debe ser un archivo comprimido, en el cual los
    # arreglos estan en dos archivos, 'videos' y 'clases'
    b = np.load(rutaDatos + dataset)
    videos = b['videos'];               clases = b['clases']
    # las clases estan desde 1..6, deben ser desde 0..5
    clases -= 1
    del b #libera memoria
    skf = StratifiedKFold(clases, n_folds=conjuntos)

    T = temporizador()
    inicio_todo = T.tic()

    contador = 0

    print("Prueba con tecnicas de validacion cruzada...")
    print("Numero de particiones: ", conjuntos)
    print("Porcentaje entrenamiento/validacion: ",porcentaje)

    costoTRN_conjunto = []; costoVAL_conjunto = []; costoTST_conjunto = []; costoTST_conjunto_final = []

    for train_index, test_index in skf:
        contador +=1
        print("Particion < " + str(contador) + " >")

        # lista de tupas, [(x_trn, y_trn), ...]
        # los datos estan normalizados...
        datos = []
        cantidad = int(len(train_index)*porcentaje)

        datosTRN = (videos[train_index[:cantidad]], clases[train_index[:cantidad]])
        datosVAL = (videos[train_index[cantidad:]],clases[train_index[cantidad:]])
        datosTST = (videos[test_index],clases[test_index])

        print("                    Clases                     :", "[c1 c2 c3 c4 c5 c6]")
        print("                                               :", "-------------------")
        print("Cantidad de clases en el conjunto Entrenamiento:", np.bincount(datosTRN[1]))
        print("Cantidad de clases en el conjunto Validacion: \t", np.bincount(datosVAL[1]))
        print("Cantidad de clases en el conjunto Test: \t", np.bincount(datosTST[1]))

        datos.append(datosTRN); datos.append(datosVAL); datos.append(datosTST)

        del datosTRN ; del datosVAL; del datosTST

        # creo la red
        ruta_kfold = rutaCompleta + carpetaConjunto + str(contador) + '/'
        os.makedirs(ruta_kfold) if not os.path.exists(ruta_kfold) else None

        # se crea la red
        clasificador = MLP(clasificacion=True, rng=None, ruta=ruta_kfold, nombre=nombre)

        # se agregan los parametros
        parametros = {'tasaAprendizaje':  tasaAprenTRN,
                      'regularizadorL1':  regularizadorL1,
                      'regularizadorL2':  regularizadorL2,
                      'momento':          momentoTRN,
                      'epocas':           epocasTRN,
                      'toleranciaError':  tolError}
        clasificador.setParametroEntrenamiento(parametros)


        # agrego tantas capas desee de regresion
        # la primera es la que indica la entrada
        # las intermedias son de regresion
        # la ultima es de clasificaicon
        for idx, _ in enumerate(capas[:-2]): # es -2 porque no debo tener en cuenta la primera ni la ultima
            clasificador.agregarCapa(unidadesEntrada=capas[idx], unidadesSalida=capas[idx+1], clasificacion=False, activacion='sigmoidea', pesos=None, biases=None)
        clasificador.agregarCapa(unidadesSalida=capas[-1], clasificacion=True, pesos=None, biases=None)

        T = temporizador()
        inicio = T.tic()
        # se entrena la red
        costoTRN, costoVAL, costoTST, costoTST_final = clasificador.entrenar(trainSet=datos[0],
                                                                             validSet=datos[1],
                                                                             testSet=datos[2],
                                                                             batch_size=tambatch,
                                                                             rapido=mlprapido)

        final = T.toc()
        print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

        costoTRN_conjunto.append(costoTRN)
        costoVAL_conjunto.append(costoVAL)
        costoTST_conjunto.append(costoTST)
        costoTST_conjunto_final.append(costoTST_final)

        # dibujar estadisticos
        clasificador.dibujarEstadisticos(mostrar=False, guardar=ruta_kfold)

        # guardando los parametros aprendidos
        clasificador.guardarObjeto(filename=nombre)

        del clasificador


    final_todo = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio_todo, final_todo)))

    print("PROMEDIO de ERRORES para los {} conjuntos".format(contador))
    print("Costo Entrenamiento: \t", np.mean(costoTRN_conjunto) * 100.)
    print("Costo Validacion: \t", np.mean(costoVAL_conjunto) * 100.)
    print("Costo Testeo: \t\t", np.mean(costoTST_conjunto) * 100.)
    print("Costo Testeo Final: \t", np.mean(costoTST_conjunto_final) * 100.)

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
