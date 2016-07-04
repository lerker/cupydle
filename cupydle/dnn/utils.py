import numpy as np
import random

import sys
import pickle
import bz2 # bzip2
import gzip # gzip

__author__ = 'lerker'


def subsample(data, size, balanced=True, seed=123):
    """

    :param data: datos a extraer un batch (ya contiene el resultado)
    :param size: tamanio del mini_bach
    :param balanced:
    :param seed:
    :return:
    """
    return data[np.random.choice(data.shape[0], size=(size,))]

def split_data(data, fractions, seed=123):
    """
    Split data en sets en base a fractions
    :param data: list or np.array
    :param fractions: list [f_train, f_valid, f_test]
    :param seed: int
    :return: sets (e.g. train, valid, test)
    """
    # Verifico que fractions sea correcto
    # TODO: assert (sum(fractions) <= 1.0, Exception("Fracciones para conjuntos incorrectas!"))

    np.random.seed(seed)
    # indice random
    indices = np.random.permutation(data.shape[0])

    # cantidad en numero para cada conjunto
    trainix = int(data.shape[0] * fractions[0])
    validationix = int(data.shape[0] * fractions[1])
    testix = int(data.shape[0] * fractions[2])

    training_idx, validation_idx, test_idx = indices[:trainix-validationix+1], \
                                             indices[trainix-validationix+1:trainix+validationix], \
                                             indices[-testix:]
    training, validation, test = data[training_idx, :], data[validation_idx, :], data[test_idx, :]

    return training, validation, test


def vectorize_label(label, n_classes):
    """
    Dada una etiqueta genera un vector de dimension (n_classes, 1) de 0 y un 1 en la posicion de la label
    e.g: si label es '5', n_classes es '8':
    return:
     v[0]=0
     v[1]=0
     v[2]=0
     v[3]=0
     v[4]=1
     v[5]=0
     v[6]=0
     v[7]=0


    :param label:
    :param n_classes:
    :return:
    """
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)



""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def save(objeto, filename=None, compression=None):
    assert filename is not None, "Nombre del archivo a guardar NO VALIDO"

    # en el filename esta la compression... la detecto
    if compression is None:
        if filename.find('.pkl') != -1: # quire decir que contiene la terminacion
            with open(filename, "wb") as f:
                pickle.dump(objeto, f)
                f.close()
        elif filename.find('.pgz') != -1: # quire decir que contiene la terminacion
            with gzip.GzipFile(filename, "w") as f:
                pickle.dump(objeto, f)
                f.close()
        elif filename.find('.pbz2') != -1:
            with bz2.BZ2File(filename, 'w') as f:
                pickle.dump(objeto, f)
                f.close()
        else:
            try:
                with open(filename + '.pkl', "wb") as f:
                    pickle.dump(objeto, f)
                    f.close()
            except:
                print("Archivo no encontrado", filename)
                sys.exit(0)
    elif compression == 'gzip':
        with gzip.GzipFile(filename + '.pgz', 'w') as f:
            pickle.dump(objeto, f)
            f.close()
    elif compression == 'bzip2':
        with bz2.BZ2File(filename + '.pbz2', 'w') as f:
            pickle.dump(objeto, f)
            f.close()
    else:
        sys.exit("Parametro de compresion no se reconoce")
    return objeto

def load(filename=None, compression=None):
    assert filename is not None, "Nombre del archivo NO VALIDO"
    objeto = None

    if compression is None:
        if filename.find('.pkl') != -1: # quire decir que contiene la terminacion
            with open(filename, "rb") as f:
                objeto = pickle.load(f)
                f.close()
        elif filename.find('.pgz') != -1: # quire decir que contiene la terminacion
            with gzip.open(filename, "rb") as f:
                objeto = pickle.load(f)
                f.close()
        elif filename.find('.pbz2') != -1:
            with bz2.open(filename, 'rb') as f:
                objeto = pickle.load(f)
                f.close()
        else:
            try:
                with open(filename + '.pkl', "rb") as f:
                    objeto = pickle.load(f)
                    f.close()
            except:
                print("Archivo no encontrado", filename)
                sys.exit(0)
    elif compression == 'gzip':
        with gzip.open(filename + '.pgz', "rb") as f:
            objeto = pickle.load(f)
            f.close()
    elif compression == 'bzip2':
        with bz2.open(filename + '.pbz2', 'rb') as f:
            objeto = pickle.load(f)
            f.close()
    else:
        sys.exit("Parametro de compresion no se reconoce")

    return objeto
