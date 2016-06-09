#!/usr/bin/python3

#from plot_mnist_utils import plot_mnist_digit
from mnist import MNIST

from mnist import save2disk
from mnist import open4disk

import numpy
import pickle

import bz2 # bzip2
import gzip # gzip


if __name__ == "__main__":

    mn = MNIST("./data")

    print(type(mn.get_training()[0]), type(mn.get_training()[1]))
    print(mn.get_training()[0].shape, mn.get_training()[1].shape)

    print(type(mn.get_testing()[0]), type(mn.get_testing()[1]))
    print(mn.get_testing()[0].shape, mn.get_testing()[1].shape)

    print(type(mn.get_validation()[0]), type(mn.get_validation()[1]))
    print(mn.get_validation()[0].shape, mn.get_validation()[1].shape)

    print(type(mn.get_training()[0][1,:]))
    print(mn.get_training()[0][1,:].shape)


    mn.info

    save2disk(mn, compresion='bzip2')

    coso = open4disk(compresion='bzip2')
    coso.info

    #print(mn.get_training()[0][0:10,:].shape)
    imagenes = mn.get_training()[0][0:10,:]

    mn.plot_one_digit(mn.get_training()[0][1,:], label=mn.get_training()[1][1], save=True)
    mn.plot_ten_digits(imagenes, crop=True, save='svg')

