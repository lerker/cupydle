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
Implementacion de una red multi-capa en GP-GPU/CPU (Theano)

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

import os
import sys
import pickle
import gzip
import time

import numpy
from numpy.random import RandomState as npRandom

import theano

from cupydle.dnn.funciones import sigmoideaTheano
from cupydle.dnn.funciones import linealRectificadaTheano
from cupydle.dnn.capas import Capa
from cupydle.dnn.capas import CapaClasificacion
from cupydle.dnn.utils import save
from cupydle.dnn.utils import load as load_utils
from cupydle.dnn.utils_theano import shared_dataset

from cupydle.dnn.stops import criterios

from cupydle.dnn.utils_theano import gpu_info
from cupydle.dnn.utils_theano import calcular_chunk

class MLP(object):
    # atributo estatico o de la clase, unico para todas las clases instanciadas
    # para acceder a el -> MLP.verbose
    # y no una instancia de el... mi_mlp.verbose
    verbose=True

    def __init__(self, clasificacion=True, rng=None, ruta='', nombre=None):

        # semilla para el random
        self.rng = (npRandom(1234) if rng is None else npRandom(rng))

        # la red es para clasificacion o regresion?
        self.clasificacion = clasificacion

        # se almacenan las layers por la que esta compuesta la red.
        self.capas = []

        # se guardan los parametros de la red a optimizar, W y b
        self.params = []

        # parametros para el entrenamiento
        self.cost   = 0.0
        self.L1     = 0.0
        self.L2_sqr = 0.0

        # se guarda la raiz del grafo para theano
        self.x = theano.tensor.matrix('x')

        # nombre del objeto, por si lo quiero diferenciar de otros en la misma
        # carpeta
        self.nombre = ('mlp' if nombre is None else nombre)

        self.parametrosEntrenamiento = {}
        self._initParametrosEntrenamiento()


        self.estadisticos = {'errorEntrenamiento':[], 'errorValidacion':[], 'errorTest':[]}

        # donde se alojan los datos
        self.ruta = ruta

    def _initParametrosEntrenamiento(self):
        """ inicializa los parametros de la red, un diccionario"""
        self.parametrosEntrenamiento['tasaAprendizaje'] = 0.0
        self.parametrosEntrenamiento['regularizadorL1'] = 0.0
        self.parametrosEntrenamiento['regularizadorL2'] = 0.0
        self.parametrosEntrenamiento['momento'] = 0.0
        self.parametrosEntrenamiento['epocas'] = 0.0
        self.parametrosEntrenamiento['activationfuntion'] = sigmoideaTheano()
        self.parametrosEntrenamiento['toleranciaError'] = 0.0
        self.parametrosEntrenamiento['tiempoMaximo'] = 0
        return 1

    def setParametroEntrenamiento(self, parametros):
        if not isinstance(parametros, dict):
            assert False, "Debe ser un diccionario"

        for key, _ in parametros.items():
            if key in self.parametrosEntrenamiento:
                self.parametrosEntrenamiento[key] = parametros[key]
            else:
                assert False, "la clave(" + str(key) + ") en la variable paramtros no existe"

        return 1

    def agregarCapa(self, unidadesSalida, clasificacion, unidadesEntrada=None,
                    activacion=sigmoideaTheano(), pesos=None, biases=None):
        """
        :type unidadesEntrada: int
        :param unidadesEntrada: cantidad de nueronas en la entrada, por defecto
                                es la cantidad de salida de la capa anterior

        :type unidadesSalida: int
        :param unidadesSalida: cantidad de neuronas en la salida

        :type activacion: Class activations
        :param activacion: funcion de activacion para la capa, Sigmoid, relu..

        :type pesos: numpy.ndarray
        :param pesos: matriz de pesos W a inicializar la capa

        :type biases: numpy.ndarray
        :param biases: matriz de biases b a inicializar la capa

        """
        # primer capa, es la entrada de mlp, osea x, para iniciar el arbol
        if not self.capas:
            entrada = self.x
        else:
            # la entrada es la salida de la ultima capa hasta ahora...
            entrada = self.capas[-1].activate()

        # las entradas deben ser las salidas de la anterior
        if unidadesEntrada is None:
            assert self.capas != [], "Unidades de entrada para esta capa?."
            unidadesEntrada =   self.capas[-1].getW().shape[1]
        else:
            if self.capas != []:    # solo si ya hay carga una capa
                assert unidadesEntrada ==   self.capas[-1].getW().shape[1]

        # si la capa no es de clasificacion, es de regresion logisctica
        if not clasificacion:
            capa = Capa(unidadesEntrada = unidadesEntrada,
                        unidadesSalida = unidadesSalida,
                        entrada = entrada,
                        rng = self.rng,
                        funcionActivacion = activacion,
                        W = pesos,
                        b = biases)
        else:
            capa = CapaClasificacion(unidadesEntrada = unidadesEntrada,
                                     unidadesSalida = unidadesSalida,
                                     entrada = entrada,
                                     W = pesos,
                                     b = biases)
        # se guardan las capas
        self.capas.append(capa)

        # se guardan los parametros de las capas (w,b) para el entrenamiento
        self.params += capa.params
        # se borra forzadamente el objeto para liberar espacio
        del capa, entrada

    def costos(self, y):
        """
        :param y: etiqueta de salida

        """
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically

        assert self.clasificacion, "Funcion solo valida para tareas de clasificacion"

        # costo, puede ser el MSE o bien el logaritmo negativo de la entropia..
        costo0 = self.capas[-1].negative_log_likelihood(y)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        costo1 = 0.0
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        costo2 = 0.0
        for capa in self.capas:
            costo1 += abs(capa.W.sum())
            costo2 += (capa.W ** 2).sum()

        self.cost   = costo0
        self.L1     = costo1
        self.L2_sqr = costo2

    def netErrors(self, y):
        return self.capas[-1].errors(y)

    def predict(self):
        assert self.clasificacion,"Funcion valida para tareas de clasificacion"
        return self.capas[-1].predict()


    def entrenar(self, trainSet, validSet, testSet, batch_size, tamMacroBatch=None):

        assert len(self.capas) != 0, "No hay capas, <<agregarCapa()>>"

        tamMiniBatch = batch_size

        y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of
                                        # [int] labels

        trainX, trainY  = shared_dataset(trainSet)
        validX, validY  = shared_dataset(validSet)
        testX, testY    = shared_dataset(testSet)

        n_train_batches = trainX.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = validX.get_value(borrow=True).shape[0] // batch_size
        n_test_batches  = testX.get_value(borrow=True).shape[0] // batch_size


        print("MEMORIA: ", gpu_info('Mb'))
        memoria_disponible = gpu_info('Mb')[0] * 0.8

        memoria_dataset = 4 * n_train_batches*batch_size/1024./1024.
        memoria_por_ejemplo = 4 * trainX.get_value(borrow=True).shape[1]/1024./1024.
        memoria_por_minibatch = memoria_por_ejemplo * tamMiniBatch

        print("memoria disponible:", memoria_disponible, "memoria dataset:", memoria_dataset, "memoria por ejemplo:", memoria_por_ejemplo, "memoria por Minibatch:", memoria_por_minibatch)

        if tamMacroBatch is None:
            tamMacroBatch = calcular_chunk(memoriaDatos=memoria_dataset, tamMiniBatch=tamMiniBatch, cantidadEjemplos=n_train_batches*batch_size)

        # necesito actualizar los costos, si no hago este paso no tengo
        # los valores requeridos
        self.costos(y)

        # actualizaciones
        updates = []
        updates = self.construirActualizaciones(costo=self.cost,
                                                actualizaciones=updates)
        costo = (self.cost +
                self.parametrosEntrenamiento['regularizadorL1'] * self.L1 +
                self.parametrosEntrenamiento['regularizadorL2'] * self.L2_sqr)

        train_model, validate_model, test_model = self.construirFunciones(
                                        datosEntrenamiento=shared_dataset(trainSet),
                                        datosValidacion=shared_dataset(validSet),
                                        datosTesteo=shared_dataset(testSet),
                                        cost=costo,
                                        batch_size=batch_size,
                                        updates=updates,
                                        y=y)


        k = self.predict()
        ll = testY
        predictor = theano.function(
                        inputs=[],
                        outputs=[k,ll],
                        givens={
                            self.x: testX},
                        name='predictor'
        )

        unidades = [self.capas[0].getW().shape[0]]
        unidades.extend([c.getB().shape[0] for c in self.capas])
        print("Entrenando un MLP, con [{}] unidades de entrada y {} unidades por capa".format(unidades[0], str(unidades[1:])))
        print("Cantidad de ejemplos para el entrenamiento supervisado: ", n_train_batches*batch_size)
        print("Tamanio del miniBatch: ", tamMiniBatch, "Tamanio MacroBatch: ", tamMacroBatch)
        print("MEMORIA antes de iterar: ", gpu_info('Mb'))
        print('Entrenando') if MLP.verbose else None

        # early-stopping parameters
        errorValidacionHistorico = numpy.inf
        epoca = 0
        looping = True
        errorTest=0.0

        iteracionesMax = criterios['iteracionesMaximas'](maxIter=self.parametrosEntrenamiento['epocas'])
        toleranciaErr = criterios['toleranciaError'](self.parametrosEntrenamiento['toleranciaError'])

        # inicio del entrenamiento por epocas
        try:
            while looping:

                epoca = epoca + 1

                # entreno con todo el conjunto de minibatches
                costoEntrenamiento = [train_model(i) for i in range(n_train_batches)]
                costoEntrenamiento = numpy.mean(costoEntrenamiento) # su media
                self.estadisticos['errorEntrenamiento'].append(costoEntrenamiento)

                errorValidacion = [validate_model(i) for i in range(n_valid_batches)]
                errorValidacion = numpy.mean(errorValidacion)
                self.estadisticos['errorValidacion'].append(errorValidacion)

                print(str('Epoca {:>3d} de {:>3d}, '
                        + 'Costo entrenamiento {:> 8.5f}, '
                        + 'Error validacion {:> 8.5f}').format(
                        epoca, self.parametrosEntrenamiento['epocas'], costoEntrenamiento*100.0, errorValidacion*100.0))

                # se obtuvo un error bajo en el conjunto validacion, se prueba en
                # el conjunto de test como funciona
                if errorValidacion < errorValidacionHistorico:
                    errorValidacionHistorico = errorValidacion
                    best_iter = epoca

                    errorTest = [test_model(i) for i in range(n_test_batches)]
                    errorTest = numpy.mean(errorTest)
                    self.estadisticos['errorTest'].append(errorTest)
                    print('|---->>Epoca {:>3d}, error test del modelo {:> 8.5f}'.format(epoca, errorTest*100.0))


                # para concatenar meter un or
                #print("1",not iteraciones(resultados=epoca))
                #print("2",not toleranciaErr(errorValidacion))

                looping = not(iteracionesMax(resultados=epoca) or toleranciaErr(errorValidacion))
            # termino el while
        except (KeyboardInterrupt, SystemExit):
            print("Se finalizo el proceso de entrenamiento. Guardando los datos")
            try:
                time.sleep(5)
            except (KeyboardInterrupt, SystemExit):
                sys.exit(1)


        print('Entrenamiento Finalizado. Mejor puntaje de validacion: {:> 8.5f} con un performance en el test de {:> 8.5f}'.format
              (errorValidacionHistorico * 100., errorTest * 100.))

        print("reales", predictor()[1][0:25])
        print("predic", predictor()[0][0:25])

    def guardarParametros(self):

        # se alamacenan los pesos y bias por cada capa
        for capa in enumerate(self.capas):
            numpy.save(self.ruta + 'PesosW' + str(capa[0]+1) + '.npy', capa[1].getW())
            numpy.save(self.ruta + 'BiasB' + str(capa[0]+1) + '.npy', capa[1].getB())

        return 0


    def construirActualizaciones(self, costo, actualizaciones):

        cost = (costo +
                self.parametrosEntrenamiento['regularizadorL1'] * self.L1 +
                self.parametrosEntrenamiento['regularizadorL2'] * self.L2_sqr)

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [theano.tensor.grad(cost, param) for param in self.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - self.parametrosEntrenamiento['tasaAprendizaje'] * gparam)
                for param, gparam in zip(self.params, gparams)
            ]

        # si ya vienen con actualizaciones (updates previas)
        #actualizaciones.append(updates)
        #actualizaciones += updates
        actualizaciones.extend(updates)
        return actualizaciones

    def construirFunciones(self, datosEntrenamiento, datosValidacion,
                           datosTesteo, cost, batch_size, updates, y):
        """
        por cuestiones de legibilidad lo pase a una funcion aparte la construccion
        de las funciones
        """

        trainX, trainY = datosEntrenamiento
        validX, validY = datosValidacion
        testX, testY = datosTesteo


        # allocate symbolic variables for the data
        index = theano.tensor.lscalar() # index to a [mini]batch

        test_model = theano.function(
                                    inputs=[index],
                                    outputs=self.netErrors(y),
                                    givens={
                                            self.x: testX[index * batch_size:(index + 1) * batch_size],
                                            y: testY[index * batch_size:(index + 1) * batch_size]
                                            },
                                    name='test_model'
        )

        validate_model = theano.function(
                                        inputs  = [index],
                                        outputs = self.netErrors(y),
                                        givens = {
                                            self.x: validX[index * batch_size:(index + 1) * batch_size],
                                            y: validY[index * batch_size:(index + 1) * batch_size]
                                        },
                                        name = 'validate_model'
        )

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
                                    inputs = [index],
                                    outputs = cost,
                                    updates = updates,
                                    givens = {
                                        self.x: trainX[index * batch_size: (index + 1) * batch_size],
                                        y: trainY[index * batch_size: (index + 1) * batch_size]
                                    },
                                    name = 'train_model'
        )
        return train_model, validate_model, test_model

if __name__ == '__main__':
    assert False, str(__file__ + " No es un modulo")
