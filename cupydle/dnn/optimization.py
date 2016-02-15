import copy
import numpy as np
import time

from cupydle.dnn.Neurons import Neurons

class GradientDescendent(object):
    """
    https://class.coursera.org/ml-003/lecture/106
    m := dataset length


    Bach Gradient Descent: use ALL 'm' examples in each iteration
    Stochastic Gradient Descent: un ONE example in each iteration
    Mini-bach Gradient Descent: use 'b' examples in each iteration

    b tipico es 10;
    b esta entre el rango [2, 100]

    la actualizacion de los pesos, es el PROMEDIO
    Thita_j = Thita_j - alfa * (1 / b) * Sum( activation_i(x) - Y)*x_i

    La ventaja esta en que el mini-bach se puede paralelizar
    osea, digamos que b=10, corro 10 jobs entrenandose cada uno, luego hago un reduce para sumar todas las actualizaciones
    y promedio las mismas, recien ahi actualizo la red principal

    si m = b el Mini-bach gradient desendent es igual a Bach Gradient descendent
    si b = 1 se transforma en 'Online Gradient Descendent', el cual actualiza los pesos para cada patron, mejores resultados en los errores, pero mas costoso de calcular
    """
    def __init__(self, model):
        self.model = copy.copy(model)

    def __update__(self, nablaw, nablab):
        #  Tener en cuenta que las correcciones son restas, por lo cual se cambia el signo
        step_w = [w * -1.0 for w in nablaw]
        step_b = [b * -1.0 for b in nablab]
        for l in range(self.model.num_layers):
            self.model.list_layers[l].update(step_w[l], step_b[l])

    def update_mini_batch(self, mini_batch, eta, momentum, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate , and ``n`` is the total size of the
        training data set."""

        nabla_b = []
        nabla_w = []
        loss_mini_batch = 0.0

        # preparo el lugar donde se almacenan los valores temporales
        for layer in self.model.list_layers:
            shape_w = layer.get_weights().shape
            shape_b = layer.get_bias().shape
            nabla_b.append(Neurons(np.zeros(shape_b), shape_b))
            nabla_w.append(Neurons(np.zeros(shape_w), shape_w))

        # primer paso, por cada patron y su salida
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b, delta_loss = self.model.__backprop__(x, y)

            # suma los delta de nabla a los nabla que ya habia.
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            # el costo de la red para el conjunto evaluado es promedio de los costos
            loss_mini_batch += delta_loss

        # update TODO fijarse si es dividido por el tamanio del batch
        nabla_w = np.multiply(nabla_w, eta / len(mini_batch))
        nabla_b = np.multiply(nabla_b, eta / len(mini_batch))

        # el promedio del batch
        loss_mini_batch /= len(mini_batch)

        # Sumo el incremento de Ws y bs
        self.__update__(nablaw=nabla_w, nablab=nabla_b)

        return nabla_w, nabla_b, loss_mini_batch

    def sgd(self, training_data, epochs, mini_batch_size, eta, momentum, valid_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        n = len(training_data)

        # TODO el random puede ir adentro o afuera, ver que comviene
        #import random.shuffle
        #random.shuffle(training_data)

        nabla_w_anterior = [w.get_weights() for w in self.model.list_layers]
        nabla_b_anterior = [b.get_bias() for b in self.model.list_layers]
        # no hay incremento de los pesos al tiempo t=0
        nabla_w_anterior = np.multiply(nabla_w_anterior, 0.0)
        nabla_b_anterior = np.multiply(nabla_b_anterior, 0.0)

        # para almacenar la evolucion del costo de la red
        loss = []

        for j in range(epochs):
            start = time.time()

            """
            nabla_w_anterior = [w.get_weights() for w in self.model.list_layers]
            nabla_b_anterior = [b.get_bias() for b in self.model.list_layers]
            # no hay incremento de los pesos al tiempo t=0
            nabla_w_anterior = np.multiply(nabla_w_anterior, 0.0)
            nabla_b_anterior = np.multiply(nabla_b_anterior, 0.0)
            """

            # TODO, aca hace una lista agrupada con las cantidades del mini_bach. posiblemente \
            # sea el algoritmo que tome un solo elemento de esa lista (deberia ser suffle) y evaluar
            """
            aca se genera una lista con los mini-bach; cada bach es de tamanio b. se generan de manera consecutiva del dataset

            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            #for mini_batch in mini_batches:
            #    nabla_w, nabla_b = self.update_mini_batch(mini_batch=mini_batch, eta=eta, momentum=momentum, n=len(training_data))
            """

            # TODO en este aproach voy a tomar al azar un conjunto de datos del dataset, de tamanio 'mini_bach'
            # puede que en algun momento toque los mismos
            mini_batch = [training_data[k] for k in np.random.randint(low=0, high=n, size=mini_batch_size)]
            nabla_w, nabla_b, costo = self.update_mini_batch(mini_batch=mini_batch, eta=eta, momentum=momentum, n=len(training_data))

            # TODO
            # termino de momento, se le suma a los pesos el multiplicativo de un estado anterior
            nabla_w_m = np.multiply(nabla_w_anterior, -1.0 * momentum) # multiplico por -1 porque el update tambien lo multiplica
            nabla_b_m = np.multiply(nabla_b_anterior, -1.0 * momentum)
            self.__update__(nablaw=nabla_w_m, nablab=nabla_b_m)
            nabla_b_anterior = nabla_b
            nabla_w_anterior = nabla_w

            hits = (self.model.evaluate(valid_data) / len(valid_data)) * 100.0
            hits = 100.0 - hits

            end = time.time()
            print("Epoch {} training complete - Error: {} [%]- Tiempo: {} [s]".format(j, round(hits, 2), round(float(end - start), 4)))

            loss.append(costo)

        return loss
