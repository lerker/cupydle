#!/usr/bin/python3

import random
import argparse

#from plot_mnist_utils import plot_mnist_digit
from mnist import MNIST



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None, type=int,
                        help="ID (position) of the letter to show")
    parser.add_argument("--training", action="store_true",
                        help="Use training set for default, WARNING!!!")
    parser.add_argument("--testing", action="store_true",
                        help="Use testing set")
    parser.add_argument("--data", default="./data",
                        help="Path to MNIST data dir")

    args = parser.parse_args()

    mn = MNIST(args.data)

    # for default always use training set if anyone is especified
    if not args.training and not args.testing:
        args.training = True

    if args.training:
        img, label = mn.load_training()
    else:
        img, label = mn.load_testing()

    if args.id:
        which = args.id
    else:
        which = random.randrange(0, len(label))

    print('Mostrarndo num: {}'.format(label[which]))
    print(mn.display(img[which]))

    mn.plot_mnist_digit(img[which], label=label[which])

    if args.training:
        print(type(mn.train_images))
        print(type(mn.train_labels))
        print(mn.train_images.shape)
        print(mn.train_labels.shape)
    else:
        print(type(mn.test_images))
        print(type(mn.test_labels))
        print(mn.test_images.shape)
        print(mn.test_labels.shape)



    import numpy

    contador = numpy.zeros((10,1),dtype=numpy.int)

    for i in range(0, len(mn.train_images)):
        contador[mn.train_labels[i]] = contador[mn.train_labels[i]] + 1

    total = len(mn.train_labels)

    for idx, val in enumerate(contador):
        print("Imagenes '"+ str(idx) + "':\t", val, "\t %:" + str(val/total*100.0) )

    """
    print(type(mn.train_images))
    print(mn.train_images.shape)

    validation_images = numpy.asarray(mn.train_images)
    validation_labels = numpy.asarray(mn.train_labels)

    cantidad_validacion = 10000
    validation_images = mn.train_images[0:cantidad_validacion]
    validation_labels = mn.train_labels[0:cantidad_validacion]

    print(validation_images.shape)
    print(validation_labels.shape)
    """

    mn.load_data()

    print(type(mn.get_training()[0]), type(mn.get_training()[1]))
    print(mn.get_training()[0].shape, mn.get_training()[1].shape)

    print(type(mn.get_testing()[0]), type(mn.get_testing()[1]))
    print(mn.get_testing()[0].shape, mn.get_testing()[1].shape)

    print(type(mn.get_validation()[0]), type(mn.get_validation()[1]))
    print(mn.get_validation()[0].shape, mn.get_validation()[1].shape)

    '''
    # esta re balanceada la base de datos, si tomo los primeros 10000
    # estan uniformemente repartidos de los digitos 0..9;
    # sino probar ;)

    ## elijo uniformemente (en el monton mas o menos elegi un poco "igual" de cada una)
    ## etiquetas al azar, y de ahi saco para formar mi conjunto de validacion
    ## osea voy tachando de aca
    #cantidad_validacion = 10000
    #labels_sel = numpy.random.random_integers(low=0, high=9, size=(cantidad_validacion,) )
    #new_train_images = numpy.asarray(mn.train_images)
    #labels_sel = labels_sel.tolist()


    new_train_labels = mn.train_labels[0:10000]
    contador2 = numpy.zeros((10,1),dtype=numpy.int)
    for i in range(0, 10000):
        contador2[new_train_labels[i]] = contador2[new_train_labels[i]] + 1

    total2 = len(new_train_labels)
    for idx, val in enumerate(contador2):
        print("Imagenes '"+ str(idx) + "':\t", val, "\t %:" + str(val/total2*100.0) )
    '''

