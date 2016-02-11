import numpy as np

import time
import json

from NeuralLayer import NeuralLayer
from NeuralLayer import ClassificationLayer
from Neurons import Neurons
import loss


class NeuralNetwork(object):
    def __init__(self, list_layers):
        self.list_layers = list_layers  # unidades 'neuronas por capa'
        self.num_layers = len(self.list_layers) - 1  # cantidad de capas

        self.loss = loss.fun_loss["MSE"]  # funcion de costo de salida
        self.loss_d = loss.fun_loss_prime["MSE"]  # funcion de costo de salida
        # self.loss = loss.fun_loss["CROSS_ENTROPY"]  # funcion de costo de salida
        # self.loss_d = loss.fun_loss_prime["CROSS_ENTROPY"]  # funcion de costo de salida
        self.evaluate_fun = loss.fun_loss['resta']

        self.clasificacion = False

        # las capas contienen las neuronas, con sus respectivos pesos y bias
        self.__init_layers__()  # inicializo las capas, los pesos

        self.hits_train = 0.0
        self.hits_valid = 0.0

    def __init_layers__(self):
        """
        Inicializacion de los pesos W y los biases b.
        Dada la lista de 'layers', el primer elemento de la misma es la entrada o capa de entrada y basicamente indica
        cuantos son los valores de entrada de la red
        # http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
        :return:
        """

        activacion = "Sigmoid"
        #activacion = "Tanh"

        # se borra el contenido de list_layers para almacenar una lista de objetos del tipo 'NeuralLayer'
        size = self.list_layers
        self.list_layers = []

        """
        for i in range(0, self.num_layers-1):
            self.list_layers.append(
                                    NeuralLayer(n_in=size[i],
                                                n_out=size[i+1],
                                                activation=activacion))
        """

        if self.clasificacion:
            self.list_layers = [NeuralLayer(n_in=x, n_out=y, activation=activacion) for x, y in
                                zip(size[:-2], size[1:-1])]
            # la ultima capa es de clasificacion, osea un softmax en vez de activacion comun
            self.list_layers.append(ClassificationLayer(n_in=size[-2], n_out=size[-1], activation=activacion))
        else:
            self.list_layers = [NeuralLayer(n_in=x, n_out=y, activation=activacion) for x, y in
                                zip(size[:-1], size[1:])]

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
            a = [None] * (
            self.num_layers + 1)  # Vector que contiene las activaciones de las salidas de cada NeuralLayer

            # a[0] = np.array(x)  # en la posicion 0 se encuentra la entrada
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
        predice la salida de una red, en el caso de ser varias salidas retorna el indice donde la misma tuvo una activacion
        mas fuerte.
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

        # TODO, este es el caso general, se debe tener varias salidas para los distintos posibles valores
        # test_results = [(np.argmax(self.__feedforward__(x)), y) for (x, y) in test_data]
        # return sum(int(x == y) for (x, y) in test_results)

        # TODO, este es el caso especial para el OR
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

        # 2 --------------------- Feed-forward
        a[0] = x
        for l in range(self.num_layers):
            (a[l + 1], d_a[l]) = self.list_layers[l].output(x=a[l], grad=True)
        # TODO a = a[1:]
        s = (len(x), 1)
        a[0] = Neurons(x, s)
        s = (len(y), 1)
        y = Neurons(y, s)

        # 3 ---------------------- Output Error
        # error de salida, delta de la ultima salida
        # el error viene dado por el error instantaneo local de cada neurona, originado por el error cuadratico
        # aca va el costo, (y_real - y_prediccion) # TODO ver el orden, antes esteba al revez. puede ser una cuestion de signos
        d_cost = a[-1] - y
        delta = d_cost

        # 4 ---------------------- Backward pass
        # desde la ultima capa hasta la primera, la ultima capa es especial el delta, lo calculo afuera
        # la formula de actualizacion siempre es la misma
        # delta_w = eta * delta * entrada_anterior
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

    def __update__(self, nablaw, nablab):
        #  Tener en cuenta que las correcciones son restas, por lo cual se cambia el signo
        step_w = [w * -1.0 for w in nablaw]
        step_b = [b * -1.0 for b in nablab]
        for l in range(self.num_layers):
            self.list_layers[l].update(step_w[l], step_b[l])

    def update_mini_batch(self, mini_batch, eta, momentum, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate , and ``n`` is the total size of the
        training data set."""

        nabla_b = []
        nabla_w = []

        # preparo el lugar donde se almacenan los valores temporales
        for layer in self.list_layers:
            shape_w = layer.get_weights().shape
            shape_b = layer.get_bias().shape
            nabla_b.append(Neurons(np.zeros(shape_b), shape_b))
            nabla_w.append(Neurons(np.zeros(shape_w), shape_w))

        # primer paso, por cada patron y su salida
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.__backprop__(x, y)

            # suma los delta de nabla a los nabla que ya habia.
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update TODO fijarse si es dividido por el tamanio del batch
        nabla_w = np.multiply(nabla_w, eta / len(mini_batch))
        nabla_b = np.multiply(nabla_b, eta / len(mini_batch))

        # TODO
        # termino de momento, se le suma a los pesos el multiplicativo de un estado anterior
        """
        nw = [w.get_weights() for w in self.list_layers]
        nb = [w.get_bias() for w in self.list_layers]
        nabla_w = np.multiply(nabla_w, momentum)
        nabla_b = np.multiply(nabla_b, momentum)
        self.__update__(nablaw=nw, nablab=nb)
        """

        # TODO todo bien hasta aca
        # Sumo el incremento de Ws y bs
        self.__update__(nablaw=nabla_w, nablab=nabla_b)

    def sgd(self, training_data, epochs, mini_batch_size, eta, momentum, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        n = len(training_data)
        for j in range(epochs):
            # TODO, aca hace una lista agrupada con las cantidades del mini_bach. posiblemente \
            # sea el algoritmo que tome un solo elemento de esa lista (deberia ser suffle) y evaluar
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, len(training_data), momentum)

            hits = (self.evaluate(test_data) / len(test_data)) * 100.0

            print("Epoch {} training complete - Hits: {}".format(j, hits))

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

        self.sgd(training_data=train, epochs=epocas, mini_batch_size=batch_size, eta=tasa_apren, momentum=momentum,
                 test_data=valid)

        hits = (self.evaluate(test) / len(test)) * 100.0
        print("Final Score {} ".format(hits))

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"layers": self.num_layers,
                # "weights": [w.tolist() for w in self.list_layers],
                # "biases": [b.tolist() for b in self.biases],
                # "cost": str(self.cost.__name__)}
                "cost":   "MSE"
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    net = NeuralNetwork(data["layers"])
    # net.weights = [np.array(w) for w in data["weights"]]
    # net.biases = [np.array(b) for b in data["biases"]]
    return net


def tiempo():
    start = time.time()
    # ... coso
    end = time.time()
    print(end - start)
