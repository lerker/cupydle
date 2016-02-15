# Libreria Internas
import numpy as np
import time
import json
import copy
import pickle as pickle

# Librerias Externas
from cupydle.dnn.NeuralLayer import NeuralLayer
from cupydle.dnn.NeuralLayer import ClassificationLayer
from cupydle.dnn.Neurons import Neurons
import cupydle.dnn.loss as loss
from cupydle.dnn.optimization import GradientDescendent


class NeuralNetwork(object):
    def __init__(self, list_layers, clasificacion=False, funcion_error="MSE", funcion_activacion="Sigmoid", w=None, b=None):
        self.list_layers = list_layers  # unidades 'neuronas por capa'
        self.num_layers = len(self.list_layers) - 1  # cantidad de capas

        # esto debe setearse
        self.clasificacion = clasificacion
        self.funcion_error = funcion_error
        self.funcion_activacion = [funcion_activacion] * self.num_layers

        self.loss = loss.fun_loss[self.funcion_error]
        self.loss_d = loss.fun_loss_prime[self.funcion_error]

        # las capas contienen las neuronas, con sus respectivos pesos y bias
        self.__init_layers__(w, b)  # inicializo las capas, los pesos

        self.hits_train = 0.0
        self.hits_valid = 0.0
    # FIN __init__

    def __init_layers__(self, w, b):
        """
        Inicializacion de los pesos W y los biases b.
        Dada la lista de 'layers', el primer elemento de la misma es la entrada o capa de entrada y basicamente indica
        cuantos son los valores de entrada de la red
        :return:
        """
        # se borra el contenido de list_layers para almacenar una lista de objetos del tipo 'NeuralLayer'
        size = self.list_layers
        self.list_layers = []

        if self.clasificacion:
            # TODO aca la activacion deberia variar con la capa, cada una puede tener distintas activaciones
            self.list_layers = [NeuralLayer(n_in=x,
                                            n_out=y,
                                            activation=self.funcion_activacion[0]) for x, y in zip(size[:-2], size[1:-1])]
            # la ultima capa es de clasificacion, osea un softmax en vez de activacion comun
            self.list_layers.append(ClassificationLayer(n_in=size[-2],
                                                        n_out=size[-1],
                                                        activation=self.funcion_activacion[0]))
        else:
            self.list_layers = [NeuralLayer(n_in=x,
                                            n_out=y,
                                            activation=self.funcion_activacion[0]) for x, y in zip(size[:-1], size[1:])]

        del size

        # si quiero cargar los w y b
        if w is not None or b is not None:
            for idx, val in enumerate(self.list_layers):
                val.set_weights(w[idx])
                val.set_bias(b[idx])
    # FIN  __init_layers__

    def __feedforward__(self, x, full_intermedio=False, grad=False):
        """
        Calcula la salida capa a capa que tiene la entrada 'x' sobre la red. Pasada hacia adelante.
        :param x: entrada
        :param full_intermedio: bandera que habilita si se quiere las salidas intermedias
        :param grad: para retornar los gradientes de cada salida por capa
        :return: lista de salidas de cada capa de la red si full_intermedio o sino la ultima
        """
        if full_intermedio:
            # Vector que contiene las activaciones de las salidas de cada NeuralLayer
            a = [None] * (self.num_layers + 1)
            a[0] = x

            for l in range(self.num_layers):
                a[l + 1] = self.list_layers[l].output(a[l])

            x = a[1:]
            del a
        else:
            for l in range(self.num_layers):
                x = self.list_layers[l].output(x)

        return x
    # FIN  __feedforward__

    def predict(self, x):
        """
        predice la salida de una red, en el caso de ser varias salidas
        retorna el indice donde la misma tuvo una activacion mas fuerte.
        :param x:
        :return:
        """
        test_result = self.__feedforward__(x)

        # me guardo el indice, ya sea de una sola salida (indice=0) o de varias
        salida = np.argmax(test_result)

        return salida

    # FIN predict

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        if self.clasificacion:
            test_results = [(np.argmax(self.__feedforward__(x).matrix), y) for (x, y) in test_data]
        else: # regresion especial
            test_results = [(int(self.__feedforward__(x).matrix > 0.5), int(y > 0.0)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # FIN evaluate

    def __backprop__(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.list_layers.weigth`` and ``self.list_layers.biases``.
        :param x: patron de entrada
        :param y: label de salida, deseada
        :return: nabla_w, nabla_b corresponden a la derivada del costo con respecto a los pesos y a los bias respectivamente
        """
        # Ver http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
        # http://neuralnetworksanddeeplearning.com/chap2.html
        # x, y: numpy array

        nabla_w = [None] * self.num_layers  # Vector que contiene los gradientes del costo respecto a W
        nabla_b = [None] * self.num_layers  # Vector que contiene los gradientes del costo respecto a b

        a = [None] * (self.num_layers + 1)  # Vector que contiene las activaciones de las salidas de cada NeuralLayer
        d_a = [None] * self.num_layers  # Vector que contiene las derivadas de las salidas activadas de cada NeuralLayer

        y = Neurons(y, (len(y), 1))

        # 2 --------------------- Feed-forward
        a[0] = Neurons(x, (len(x), 1))
        for l in range(self.num_layers):
            (a[l + 1], d_a[l]) = self.list_layers[l].output(x=a[l], grad=True)

        # 3 ---------------------- Output Error
        # error de salida, delta de la ultima salida
        # el error viene dado por el error instantaneo local de cada neurona, originado por el error cuadratico
        # aca va el costo, derivada de la funcion, en MSE => (y_prediccion - real)
        # d_cost = a[-1] - y
        #d_cost = Neurons(self.loss_d(y.matrix, a[-1].matrix), y.shape) # TODO segunda opcion
        d_cost = Neurons(self.loss_d(y.matrix, a[-1].matrix), a[-1].shape) # TODO segunda opcion
        delta = d_cost

        # 4 ---------------------- Backward pass
        # desde la ultima capa hasta la primera, la ultima capa es especial el delta, lo calculo afuera
        # la formula de actualizacion siempre es la misma
        # delta_w = eta * delta * entrada_anterior
        # cuando el problema es de clasificacion, en la ultima capa se usa un 'softmax', y si la salida del error es de
        # 'cross_entropy', no se debe mltiplicar el delta por la derivada de la funcion de activacion. SOLO en el caso de
        # REGRESION
        #if not self.clasificacion:
        if not self.clasificacion and self.funcion_error == "CROSS_ENTROPY":
            delta = delta.mul_elemwise(d_a[-1])

        nabla_w[-1] = delta.outer(a[-2])
        nabla_b[-1] = delta

        # Note that the variable l
        # l = 1 means the last layer of neurons, l = 2 is the second-last layer, and so on.

        for l in range(2, self.num_layers + 1):
            w_t = self.list_layers[-l+1].get_weights().transpose()
            tmp = w_t.mul_array(delta)
            delta = d_a[-l].mul_elemwise(tmp)
            nabla_b[-l] = delta
            nabla_w[-l] = delta.outer(a[-l-1])

        return nabla_w, nabla_b


    def fit(self, train, valid, test, epocas, tasa_apren, momentum, batch_size=1):
        """
        La idea es que se tiene el conjunto de datos, se puede operar de dos formas,
        modo batch,
            se calcula el error de toda la red para todos los datos, (promedio del todal)
                Schotastic Gradient Descendent
                Gradient Descendent
                    minimizan la funcion de error (error cuadratico por ejemplo)
            se propaga hacia atras el erro con backprop
        modo online,
            se calcula el error ejemplo a ejemplo
            se propaga hacia atras el error
        """
        start = time.time()

        G = GradientDescendent(self)
        G.sgd(training_data=train, epochs=epocas, mini_batch_size=batch_size, eta=tasa_apren, momentum=momentum, valid_data=valid)

        end = time.time()
        print("Tiempo total requerido: {} [s]".format(round(float(end - start), 4)))

        hits = (self.evaluate(test) / len(test)) * 100.0
        print("Final Score {} ".format(hits))

    def save(self, filename, path=''):
        """Save the neural network to the file ``filename``."""
        data = {"layers": [(l.n_out, l.n_in) for l in self.list_layers],
                "weights": [w.get_weights().matrix.tolist() for w in self.list_layers],
                "biases": [b.get_bias().matrix.tolist() for b in self.list_layers],
                # "cost": str(self.cost.__name__)}
                "cost":   str(self.funcion_error)
                }
        f = open(path + filename + '.cupydle', "w")
        json.dump(data, f)
        f.close()

    def save_weights(self, filename, path):
        """
        Save de Network Weights to 'filename'
        :param filename:
        :param path:
        :return:
        """
        file = open(path+filename+'.cupydle', 'w')
        pickler = pickle.Pickler(file, -1)
        pickler.dump(self)
        file.close()
        return


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    capas = [data["layers"][0][1]]  # entrada de la red, se duplica en la lista para mantener consistencia
    capas_tmp = [l[0] for l in data["layers"]]
    capas.extend(capas_tmp)
    net = NeuralNetwork(list_layers=capas, w=data["weights"], b=data["biases"])
    return net


def tiempo():
    start = time.time()
    # ... coso
    end = time.time()
    print(end - start)
