__author__ = "Nelson Ponzoni"
__copyright__ = "Copyright 2015-2016, Proyecto Final de Carrera"
__credits__ = ["Nelson Ponzoni"]
__license__ = "GPL"
__version__ = "20160101"
__maintainer__ = "Nelson Ponzoni"
__email__ = "npcuadra@gmail.com"
__status__ = "Production"

"""

"""
# Dependencias externas
import numpy as np

# Dependencias internas
from Neurons import Neurons
from Stops import Criterion

# Librerias de Python
import copy

class OptimizerParameters:
    def __init__(self, algorithm='Adadelta', options=None, stops=None,
                 merge_criter='w_avg', merge_goal='hits'):
        if options is None:  # Agrego valores por defecto
            if algorithm == 'Adadelta':
                options = {'step-rate': 1, 'decay': 0.99, 'momentum': 0.0, 'offset': 1e-8}
            elif algorithm == 'GD':
                options = {'step-rate': 1, 'momentum': 0.3, 'momentum_type': 'standart'}
        self.options = options
        self.algorithm = algorithm
        if stops is None:
            stops = [Criterion['MaxIterations'](10),
                     Criterion['AchieveTolerance'](0.99, key='hits')]
        self.stops = stops
        self.merge = {'criter': merge_criter, 'goal': merge_goal}



class Optimizer(object):
    """

    """

    def __init__(self, model, data, parameters=None):
        self.model = copy.copy(model)
        self.num_layers = model.num_layers
        self.data = data
        if parameters is None:
            parameters = OptimizerParameters()
        self.parameters = parameters
        self.cost = 0.0
        self.n_iter = 0
        self.hits = 0.0
        self.step_w = None
        self.step_b = None

    def _iterate(self):
        # Implementacion hecha en las clases que heredan
        yield

    def _update(self):
        #  Tener en cuenta que las correcciones son restas, por lo cual se cambia el signo
        self.step_w = [w * -1.0 for w in self.step_w]
        self.step_b = [b * -1.0 for b in self.step_b]
        self.model.update(self.step_w, self.step_b)

    def results(self):
        return {
            'model': self.model.list_layers,
            'hits': self.hits,
            'iterations': self.n_iter,
            'cost': self.cost
        }

    def __iter__(self):
        results = None
        for stop in self._iterate():
            results = self.results()
        yield results

    def check_stop(self, check_all=False):
        if check_all is True:
            stop = all(c(self.results()) for c in self.parameters.stops)
        else:
            stop = any(c(self.results()) for c in self.parameters.stops)
        return stop

class GD(Optimizer):
    def __init__(self, model, data, parameters=None):
        super(GD, self).__init__(model, data, parameters)
        self._init_acummulators()

    def _init_acummulators(self):
        """
        Inicializo acumuladores usados para la optimizacion
        :return:
        """
        self.step_w = []
        self.step_b = []
        for layer in self.model.list_layers:
            shape_w = layer.get_weights().shape
            shape_b = layer.get_bias().shape
            self.step_w.append(Neurons(np.zeros(shape_w), shape_w))
            self.step_b.append(Neurons(np.zeros(shape_b), shape_b))

    def _iterate(self):
        while self.check_stop() is False:
            m = 1.0
            sr = 1
            # --- Entrenamiento ---
            for lp in self.data:  # Por cada LabeledPoint del conj de datos
                for l in range(self.num_layers):
                    cost = 0.0
                    # ['momentum_type'] == 'standard':
                    # Computar el gradiente
                    cost, (nabla_w, nabla_b) = self.model.cost(lp.features, lp.label)
                    self.step_w[l] = nabla_w[l] * sr + self.step_w[l] * m
                    self.step_b[l] = nabla_b[l] * sr + self.step_b[l] * m
                # Aplicar actualizaciones a todas las capas
                self.cost = cost
                self._update()
            # --- Error de clasificacion---
            data = copy.deepcopy(self.data)
            self.hits = self.model.evaluate(data)
            self.n_iter += 1
            yield self.check_stop()

Minimizer = {'GD': GD}

# Funciones usadas en model


def optimize(model, data, mini_batch=50, params=None, seed=123):
    final = {
        'model': model.list_layers,
        'hits': 0.0,
        'epochs': 0,
        'cost': -1.0,
        'seed': seed
    }
    # TODO: modificar el batch cada tantas iteraciones (que no sea siempre el mismo)

    batch = data
    minimizer = Minimizer[params.algorithm](model, batch, params)
    # TODO: OJO que al ser un iterator, result vuelve a iterar cada vez
    # que se hace una accion desde la funcion 'train' (solucionado con .cache() para que no se vuelva a lanzar la task)
    for result in minimizer:
        final = result
        print('Cant de iteraciones: ' + str(result['iterations']) +\
              '. Hits en batch: ' + str(result['hits']) + \
              '. Costo: ' + str(result['cost']))
    final['seed'] = seed
    return final


def mix_models(left, right):
    """
    Se devuelve el resultado de sumar las NeuralLayers
    de left y right
    :param left: list of NeuralLayer
    :param right: list of NeuralLayer
    :return: list of NeuralLayer
    """
    for l in range(len(left)):
        w = right[l].get_weights()
        b = right[l].get_bias()
        left[l].update(w, b)  # Update suma el w y el b
    return left

