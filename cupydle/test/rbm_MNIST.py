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
Implementacion de una Maquina de Boltmann Restringida en GP-GPU/CPU (Theano)
Aplicado a la base de datos de 'benchmark' MNIST

http://yann.lecun.com/exdb/mnist/

# version corta
optirun python3 cupydle/test/rbm_MNIST.py --directorio "test_RBM_MNIST" --dataset "mnist_minmax.npz" --batchsize 10 -lrW 0.01 -lrV 0.01 -lrO 0.01 -wc 0.0 --momentoTRN 0.0 --dropout 1.0 --lepocaTRN 50 --visibles 784 --ocultas 10 -pcd --gibbs 1 --tipo 'binaria'

# version larga
optirun python3 cupydle/test/rbm_MNIST.py --directorio "test_RBM_MNIST" --dataset "mnist_minmax.npz"  --lepocaTRN 50 --visibles 784 --ocultas 10 -pcd
"""


# dependencias internas
import os, argparse, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.rbm import BRBM, GRBM


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prueba de una RBM sobre MNIST')

    parser.add_argument('--directorio',       type=str,   dest="directorio",     default='test_RBM', required=False, help="Carpeta donde se almacena la corrida actual")
    parser.add_argument('--nombre',           type=str,   dest="nombre",         default='rbm',      required=False, help="Nombre del modelo")
    parser.add_argument('--dataset',          type=str,   dest="dataset",        default=None,       required=True,  help="Archivo donde esta el dataset, [videos, clases].npz")
    parser.add_argument('-let', '--lepocaTRN',type=int,   dest="epocasTRN",      default=10,         required=False, help="cantidad de epocas de entrenamiento")
    parser.add_argument('-b', '--batchsize',  type=int,   dest="tambatch",       default=10,         required=False, help="Tamanio del minibatch para el entrenamiento")
    parser.add_argument('-lrW',               type=float, dest="tasaAprenW",     default=0.01,       required=False, help="Tasa de aprendizaje para los pesos")
    parser.add_argument('-lrV',               type=float, dest="tasaAprenV",     default=0.01,       required=False, help="Tasa de aprendizaje para los bias visibles")
    parser.add_argument('-lrO',               type=float, dest="tasaAprenO",     default=0.01,       required=False, help="Tasa de aprendizaje para los bias ocultos")
    parser.add_argument('-wc',                type=float, dest="weightcost",     default=0.00,       required=False, help="Tasa de castigo a los pesos (weieght cost)")
    parser.add_argument('--momentoTRN',       type=float, dest="momentoTRN",     default=0.0,        required=False, help="Tasa de momento para la etapa de entrenamiento")
    parser.add_argument('--dropout',          type=float, dest="dropout",        default=1.0,        required=False, help="Tasa dropout para las unidades ocultas, p porcentaje de activacion")
    parser.add_argument('-v', '--visibles',   type=int,   dest="visibles",       default=784,        required=True,  help="Cantidad de Unidades Visibles")
    parser.add_argument('-o', '--ocultas',    type=int,   dest="ocultas",        default=100,        required=True,  help="Cantidad de Unidades Ocultas")
    parser.add_argument('-pcd',               action="store_true",  dest="pcd",  default=False,      required=False, help="Activa el entrenamiento con Divergencia Contrastiva Persistente")
    parser.add_argument('-g', '--gibbs',      type=int,   dest="pasosGibbs",     default=1,          required=False, help="Cantidad de pasos de Gibbs para la Divergencia Contrastiva (Persistente?)")
    parser.add_argument('--tipo',             type=str,   dest="tipo",           default='binaria',  required=False, help="Tipo de de RBM (binaria, gaussiana)")
    argumentos = parser.parse_args()

    # parametros pasados por consola
    directorio      = argumentos.directorio
    nombre          = argumentos.nombre
    dataset         = argumentos.dataset
    epocasTRN       = argumentos.epocasTRN
    tambatch        = argumentos.tambatch
    tasaAprenW      = argumentos.tasaAprenW
    tasaAprenV      = argumentos.tasaAprenV
    tasaAprenO      = argumentos.tasaAprenO
    momentoTRN      = argumentos.momentoTRN
    visibles        = argumentos.visibles
    ocultas         = argumentos.ocultas
    pcd             = argumentos.pcd
    pasosGibbs      = argumentos.pasosGibbs
    weightcost      = argumentos.weightcost
    dropout         = argumentos.dropout
    tipo            = argumentos.tipo

    # chequeos
    tasaAprenW      = np.float32(tasaAprenW)
    tasaAprenV      = np.float32(tasaAprenV)
    tasaAprenO      = np.float32(tasaAprenO)
    momentoTRN      = np.float32(momentoTRN)
    weightcost      = np.float32(weightcost)
    dropout         = np.float32(dropout)
    assert tipo == 'binaria' or tipo == 'gaussiana', "Tipo de RBM no implementada"

    # configuraciones con respecto a los directorios
    directorioActual= os.getcwd()                                  # directorio actual de ejecucion
    rutaTest        = directorioActual + '/cupydle/test/mnist/'     # sobre el de ejecucion la ruta a los tests
    rutaDatos       = directorioActual + '/cupydle/data/DB_mnist/'  # donde se almacenan la base de datos
    carpetaTest     = directorio + '/'                   # carpeta a crear para los tests
    rutaCompleta    = rutaTest + carpetaTest

    os.makedirs(rutaCompleta) if not os.path.exists(rutaCompleta) else None

     # carga los datos
    datos = np.load(rutaDatos + dataset)

    entrenamiento        = datos['entrenamiento'].astype(np.float32)
    entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
    validacion           = datos['validacion'].astype(np.float32)
    validacion_clases    = datos['validacion_clases'].astype(np.int32)
    testeo               = datos['testeo'].astype(np.float32)
    testeo_clases        = datos['testeo_clases'].astype(np.int32)
    del datos # libera memoria

    datos = []
    datos.append((entrenamiento, entrenamiento_clases))
    datos.append((validacion, validacion_clases))
    datos.append((testeo, testeo_clases))
    del entrenamiento, entrenamiento_clases, validacion, validacion_clases
    del testeo, testeo_clases

    print("                    Clases                     :", "[1     2    3    4    5    6    7    8    9    10]")
    print("                                               :", "--------------------------------------------------")
    print("Cantidad de clases en el conjunto Entrenamiento:", np.bincount(datos[0][1]))
    print("Cantidad de clases en el conjunto Validacion: \t", np.bincount(datos[1][1]))

    # creo la red
    if tipo == "binaria":
        red = BRBM(n_visible=visibles, n_hidden=ocultas, nombre=nombre, ruta=rutaCompleta)
    elif tipo == 'gaussiana':
        red = GRBM(n_visible=visibles, n_hidden=ocultas, nombre=nombre, ruta=rutaCompleta)
    else:
        raise NotImplementedError('RBM no implementada')

    parametros={'lr_pesos': tasaAprenW,
                'lr_bvis':  tasaAprenV,
                'lr_bocu':  tasaAprenO,
                'momento':  momentoTRN,
                'costo_w':  weightcost,
                'dropout':  dropout,    # probabilidad de actividad en la neurona, =1 todas, =0 ninguna
                'epocas':   epocasTRN}

    red.setParametros(parametros)

    T = temporizador()
    inicio = T.tic()

    red.entrenamiento(data=np.concatenate((datos[0][0],datos[1][0]), axis=0),
                      tamMiniBatch=tambatch,
                      tamMacroBatch=None,
                      pcd=pcd,
                      gibbsSteps=pasosGibbs,
                      filtros=True)

    final = T.toc()
    print("Tiempo total para entrenamiento: {}".format(T.transcurrido(inicio, final)))

    print('Guardando el modelo en ...', rutaCompleta)
    inicio = T.tic()
    red.guardarModelo(nombreArchivo="RBM_MNIST")
    final = T.toc()
    print("Tiempo total para guardar: {}".format(T.transcurrido(inicio, final)))

else:
    assert False, "Esto no es un modulo, es un TEST!!!"
