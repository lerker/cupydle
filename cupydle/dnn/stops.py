#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
criterios de paradas para el entrenamiento..
"""

# Librerias de Python
import signal
import sys
import time

class iteracionesMaximas(object):

    def __init__(self, maxIter):
        self.maxIter = maxIter

    def __call__(self, resultados):
        retorno = False
        if isinstance(resultados, dict):
            retorno = (resultados['epocas'] >= self.maxIter)
        else:
            retorno = (resultados >= self.maxIter)
        return retorno


class toleranciaError(object):

    def __init__(self, tolerancia):
        self.tolerancia = tolerancia

    def __call__(self, resultados):
        retorno = False
        if isinstance(resultados, dict):
            retorno = (resultados['error'] <= self.tolerancia)
        else:
            retorno = (resultados <= self.tolerancia)
        return retorno

class tiempoTranscurrido(object):

    def __init__(self, tiempoMaximo):
        self.tiempoInicial = time.time()
        self.tiempoMaximo = tiempoMaximo

    def __call__(self, resultados):
        retorno = False
        if isinstance(resultados, dict):
            retorno = (resultados['tiempo'] - self.tiempoInicial > self.tiempoMaximo)
        else:
            retorno = (resultados - self.tiempoInicial > self.tiempoMaximo)
        return retorno


class noMejorQueAntesError(object):

    def __init__(self):
        self.antes = numpy.inf

    def __call__(self, resultados):
        retorno = False

        if isinstance(resultados, dict):
            if resultados['error'] < self.antes: #menor error ahora actualizar
                self.antes = resultados['error']
                retorno = True
        else:
            if resultados < self.antes: #menor error ahora actualizar
                self.antes = resultados
                retorno = True

        return retorno


# TODO
class Patience(object):
    """
    Stop criterion inspired by Bengio's patience method.
    The idea is to increase the number of iterations until stopping by
    a multiplicative and/or additive constant once a new best candidate is
    found.
    Attributes
    ----------
    func_or_key : function, hashable
        Either a function or a hashable object. In the first case, the function
        will be called to get the latest loss. In the second case, the loss
        will be obtained from the in the corresponding field of the ``info``
        dictionary.
    initial : int
        Initial patience. Lower bound on the number of iterations.
    grow_factor : float
        Everytime we find a sufficiently better candidate (determined by
        ``threshold``) we increase the patience multiplicatively by
        ``grow_factor``.
    grow_offset : float
        Everytime we find a sufficiently better candidate (determined by
        ``threshold``) we increase the patience additively by ``grow_offset``.
    threshold : float, optional, default: 1e-4
        A loss of a is assumed to be a better candidate than b, if a is larger
        than b by a margin of ``threshold``.
    """

    def __init__(self, initial, key='hits', grow_factor=1., grow_offset=0.,
                 threshold=1e-4):
        if grow_factor == 1 and grow_offset == 0:
            raise ValueError('need to specify either grow_factor != 1'
                             'or grow_offset != 0')
        self.key = key
        self.patience = initial
        self.grow_factor = grow_factor
        self.grow_offset = grow_offset
        self.threshold = threshold

        self.best_value = float('inf')

    def __call__(self, results):
        i = results['error']
        value = results[self.key]
        if value > self.best_value:
            if (value - self.best_value) > self.threshold and i > 0:
                self.patience = max(i * self.grow_factor + self.grow_offset,
                                    self.patience)
            self.best_value = value

        return i >= self.patience


class controlDelTeclado:
    """
    killer = controlDelTeclado()
    while True:
        time.sleep(1)
        print("doing something in a loop ...")
        if killer():
            print("matando")
            break

    print("End of the program. I was killed gracefully :)")
    """

    # OPCION 2 con doble manejo de teclas, try anidados
    # TODO implementar con decoradores
    # funcion de dos parametos, el primero la ejecucion normal
    # la segunda el guardado
    # si se acciona dos veces ctrl+c se corta
    """
    import time
    import sys
    def programa():
        print("soy el programa")

    def guardando():
        print("estoy guardando")


    while True:
        try:
            time.sleep(1)
            programa()
        except (KeyboardInterrupt, SystemExit):
            print("lo cortaste")
            while True:
                try:
                    time.sleep(1)
                    guardando()
                except (KeyboardInterrupt, SystemExit):
                    print("Saliendo YAA")
                    sys.exit(1)
            #raise
        except:
            pass
    """

    matar = False

    def __init__(self):
        # cuando se crea la clase, lo que hace es cambiar la funcion a la que
        # se llama cuando se apretan las teclas. En este caso 'matar'
        signal.signal(signal.SIGINT, self.matar)
        signal.signal(signal.SIGTERM, self.matar)

    def __call__(self):
        return self.matar

    def matar(self, signum, frame):
        self.matar = True

# diccionario a importar, del cual se llaman a las funciones.
criterios = {'iteracionesMaximas':  iteracionesMaximas,
            'toleranciaError':      toleranciaError,
            'tiempoTranscurrido':   tiempoTranscurrido,
            'noMejorQueAntesError': noMejorQueAntesError,
            'Patience':             Patience,
            'controlDelTeclado':    controlDelTeclado}


if __name__ == '__main__':
    raise ImportError(str(__file__ + " No es un modulo!!!"))
