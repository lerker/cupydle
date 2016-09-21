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
optirun python3 cupydle/test/rbm_FACE.py --directorio "test_RBM" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -p 0.8 -b 10 -lrW 0.01 --lepocaTRN 50 --visibles 85 --ocultas 6 --gibbs 1

"""

# dependencias internas
import os, argparse, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.rbm import RBM

# TODO ELIMINAR
from cupydle.dnn.unidades import UnidadBinaria
from cupydle.dnn.unidades import UnidadGaussiana


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre KLM')

    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_RBM', required=False, help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--nombre',           type=str,   dest="nombre",         default='rbm',      required=False, help="Nombre del modelo")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-let', '--lepocaTRN',type=int,   dest="epocasTRN",      default=10,         required=False, help="cantidad de epocas de entrenamiento")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-p', '--porcentaje', type=float, dest="porcentaje",     default=0.8,        required=False, help="Porcentaje en que el conjunto de entrenamiento se detina para entrenar y testeo")
    parser.add_argument('-lrW',               type=float, dest="tasaAprenW",     default=0.01,       required=False, help="Tasa de aprendizaje para los pesos")
    parser.add_argument('-lrV',               type=float, dest="tasaAprenV",     default=0.01,       required=False, help="Tasa de aprendizaje para los bias visibles")
    parser.add_argument('-lrO',               type=float, dest="tasaAprenO",     default=0.01,       required=False, help="Tasa de aprendizaje para los bias ocultos")
    parser.add_argument('-wc',                type=float, dest="weightcost",     default=0.00,       required=False, help="Tasa de castigo a los pesos (weieght cost)")
    parser.add_argument('--momentoTRN',       type=float, dest="momentoTRN",     default=0.0,        required=False, help="Tasa de momento para la etapa de entrenamiento")
    parser.add_argument('--dropVis',          type=float, dest="dropVis",        default=1.0,        required=False, help="Tasa dropout para las unidades visibles, p porcentaje de activacion")
    parser.add_argument('--dropOcu',          type=float, dest="dropOcu",        default=1.0,        required=False, help="Tasa dropout para las unidades ocultas, p porcentaje de activacion")
    parser.add_argument('-v', '--visibles',   type=int,   dest="visibles",       default=230300,     required=True,  help="Cantidad de Unidades Visibles")
    parser.add_argument('-o', '--ocultas',    type=int,   dest="ocultas",        default=100,        required=True,  help="Cantidad de Unidades Ocultas")
    parser.add_argument('-pcd',               action="store_true",  dest="pcd",  default=False,      required=False, help="Activa el entrenamiento con Divergencia Contrastiva Persistente")
    parser.add_argument('-g', '--gibbs',      type=int,   dest="pasosGibbs",     default=1,          required=False, help="Cantidad de pasos de Gibbs para la Divergencia Contrastiva (Persistente?)")
    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio      = argumentos.directorio
    nombre          = argumentos.nombre
    dataset         = argumentos.dataset
    epocasTRN       = argumentos.epocasTRN
    tambatch        = argumentos.tambatch
    porcentaje      = argumentos.porcentaje
    tasaAprenW      = argumentos.tasaAprenW
    tasaAprenV      = argumentos.tasaAprenV
    tasaAprenO      = argumentos.tasaAprenO
    momentoTRN      = argumentos.momentoTRN
    visibles        = argumentos.visibles
    ocultas         = argumentos.ocultas
    pcd             = argumentos.pcd
    pasosGibbs      = argumentos.pasosGibbs
    weightcost      = argumentos.weightcost
    dropVis         = argumentos.dropVis
    dropOcu         = argumentos.dropOcu

    # chequeos
    tasaAprenW      = np.float32(tasaAprenW)
    tasaAprenV      = np.float32(tasaAprenV)
    tasaAprenO      = np.float32(tasaAprenO)
    momentoTRN      = np.float32(momentoTRN)
    weightcost      = np.float32(weightcost)
    dropVis         = np.float32(dropVis)
    dropOcu         = np.float32(dropOcu)
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

    # lista de tupas, [(x_trn, y_trn), ...]
    # los datos estan normalizados...
    cantidad = int(clases.shape[0] * porcentaje)
    # la cantidad de ejemplos debe ser X partes enteras del minibatch, para que el algoritmo tome de a cachos y procese
    # es por ello que acomodo la cantidad hasta que quepa
    while (cantidad % tambatch):
        cantidad += 1
        assert cantidad != clases.shape[0], "Porcentaje trn/test muy alto, disminuir"

    datos = []
    datos = [(videos[:cantidad], clases[:cantidad]), (videos[cantidad:],clases[cantidad:])]

    print("Porcentaje entrenamiento/validacion: ",porcentaje)
    print("                    Clases                     :", "[c1 c2 c3 c4 c5 c6]")
    print("                                               :", "-------------------")
    print("Cantidad de clases en el conjunto Entrenamiento:", np.bincount(datos[0][1]))
    print("Cantidad de clases en el conjunto Validacion: \t", np.bincount(datos[1][1]))

    # creo la red
    red = RBM(n_visible=visibles, n_hidden=ocultas, nombre=nombre, ruta=rutaCompleta)

    parametros = {'lr_pesos':       tasaAprenW,
                'lr_bvis':          tasaAprenV,
                'lr_bocu':          tasaAprenO,
                'momento':          momentoTRN,
                'costo_w':          weightcost,
                'unidadesVisibles': UnidadBinaria(),
                'unidadesOcultas':  UnidadBinaria(),
                'dropoutVisibles':  dropVis, # probabilidad de actividad en la neurona, =1 todas, =0 ninguna
                'dropoutOcultas':   dropOcu} # probabilidad de actividad en la neurona, =1 todas, =0 ninguna


    #red.setParams(parametros)
    #red.setParams({'epocas':epocasTRN})
    red.setParametros(parametros)
    red.setParametros({'epocas':epocasTRN})

    T = temporizador()
    inicio = T.tic()

    red.entrenamiento(data=datos[0][0],
                      tamMiniBatch=tambatch,
                      tamMacroBatch=None,
                      pcd=pcd,
                      gibbsSteps=pasosGibbs,
                      validationData=None,
                      filtros=True)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    print('Guardando el modelo en ...', rutaCompleta)
    inicio = T.tic()
    red.guardar(nombreArchivo="RBM_KLM")
    final = T.toc()
    print("Tiempo total para guardar: {}".format(T.transcurrido(inicio, final)))

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
