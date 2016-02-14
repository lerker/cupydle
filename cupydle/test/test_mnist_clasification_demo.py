import nose


def test_mnist_clasification_demo():
    # Dependencias Internas
    import numpy as np

    # Dependencias Externas
    from cupydle.dnn.NeuralNetwork import NeuralNetwork
    from cupydle.test.mnist_loader import MNIST as mnist_loader

    def label_to_vector(label, n_classes):
        lab = np.zeros((n_classes, 1), dtype=np.int8)
        label = int(label)
        lab[label] = 1
        return np.array(lab).transpose()

    # Seteo de los parametros
    capa_entrada = 784
    capa_oculta_1 = 30
    capa_salida = 10
    capas = [capa_entrada, capa_oculta_1, capa_salida]

    mnist = mnist_loader(path="cupydle/data")
    training_data_x, training_data_y = mnist.load_training()

    x_tr = np.array(training_data_x, dtype=float)
    y_tr = np.reshape(np.array(training_data_y, dtype=float), (len(training_data_y), 1))
    y_tr = [label_to_vector(y, 10)[0] for y in y_tr]
    training_data = [(x, y) for x, y in zip(x_tr, y_tr)]

    testing_data_x, testing_data_y = mnist.load_testing()
    x_ts = np.array(testing_data_x, dtype=float)
    y_ts = np.reshape(np.array(testing_data_y, dtype=float), (len(testing_data_y), 1))
    datos_tst = [(x, y) for x, y in zip(x_ts, y_ts)]

    tmp = training_data
    cantidad = len(tmp)
    porcentaje = 80
    datos_trn = tmp[0: int(cantidad * porcentaje / 100)]
    datos_vld = tmp[int(cantidad * porcentaje / 100 + 1):]
    print("Training Data size: " + str(len(datos_trn)))
    print("Validating Data size: " + str(len(datos_vld)))
    print("Testing Data size: " + str(len(datos_tst)))

    net = NeuralNetwork(list_layers=capas, clasificacion=True, funcion_error="CROSS_ENTROPY",
                        funcion_activacion="Sigmoid", w=None, b=None)

    net.fit(train=datos_trn, valid=datos_vld, test=datos_tst, batch_size=100, epocas=20, tasa_apren=0.2, momentum=0.1)
    net.save('mnist_demo')
    """
    /usr/bin/python3.5/run/media/lerker/Documentos/Proyecto/Codigo/cupydle/cupydle/test/test_mnist_clasification_demo.py
    Training Data size: 48000
    Validating Data size: 11999
    Testing Data size: 10000
    Epoch 0 training complete - Error: 29.34
    Epoch 1 training complete - Error: 31.02
    Epoch 2 training complete - Error: 24.78
    Epoch 3 training complete - Error: 24.5
    Epoch 4 training complete - Error: 32.34
    Epoch 5 training complete - Error: 23.41
    Epoch 6 training complete - Error: 25.84
    Epoch 7 training complete - Error: 19.82
    Epoch 8 training complete - Error: 20.62
    Epoch 9 training complete - Error: 20.75
    Epoch 10 training complete - Error: 22.27
    Epoch 11 training complete - Error: 20.23
    Epoch 12 training complete - Error: 20.63
    Epoch 13 training complete - Error: 22.87
    Epoch 14 training complete - Error: 19.48
    Epoch 15 training complete - Error: 16.83
    Epoch 16 training complete - Error: 16.46
    Epoch 17 training complete - Error: 18.25
    Epoch 18 training complete - Error: 17.13
    Epoch 19 training complete - Error: 16.29
    Final Score 83.71

    Process finished with exit code 0
    """


if __name__ == '__main__':
    nose.run(defaultTest=__name__)
