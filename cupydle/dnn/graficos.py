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
modulo de graficos,
filtros, visualizaciones varias
"""


""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy
import os
import matplotlib.pylab as plt

def display_avalible():
    """
    si se ejecuta en un servidor, retorna falso... (si no hay pantalla... )
    util para el show del ploteo
    """
    return ('DISPLAY' in os.environ)



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

def filtrosConstructor(images, titulo, formaFiltro, nombreArchivo=None, mostrar=False, forzar=False):
        # la forma de la imagen es inferida, debe ser cuadrada
        # para el mnist es de 28,28 en la primer capa..
        # si no es cuadra (cantidad de unidades no tiene raiz cuadrada exacta..)
        # debe fallar, solo a que fuerce el plot

        # TODO hace lo de formaImagen adaptativo... en caso de que no sea cuadrada perfecto agarrar lo que corresponda

        entrada = images.shape[1] # 784 mnist
        # cuadrado perfecto?
        check = int(numpy.floor(numpy.mod(entrada,numpy.floor(numpy.power(entrada,1/2))**2)))
        assert not (check!=0 and forzar), "Forma de la imagen no corresponde"
        #tmp = int(numpy.floor(numpy.mod(entrada,numpy.floor(numpy.power(entrada,1/2))**2)))
        #formaImagen = (tmp, tmp)

        formaImagen = (int(numpy.power(entrada,1/2)) ,int(numpy.power(entrada,1/2)))


        n_col, n_row = formaFiltro

        #plt.figure(figsize=(2. * n_col, 2. * n_row))
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close("all") # Close a figure window
        plt.figure(figsize=(12,12))
        plt.title(titulo, size=16)
        fig = plt.gcf()
        DPI = fig.get_dpi()
        fig.set_size_inches(1024.0/float(DPI),1024.0/float(DPI))
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            plt.gca().set_aspect('equal', adjustable='box')

            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(formaImagen), cmap=plt.cm.gray,
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        #matplotlib.pyplot.tight_layout()
        #plt.subplots_adjust(hspace = .001, wspace=.001, left=0.001)
        #plt.subplots_adjust(wspace=.01, hspace=.01)
        plt.subplots_adjust(0,0,1,1,0,0)
        if nombreArchivo is not None:
            #print("guardando los filtros en: " + nombreArchivo)
            plt.savefig(nombreArchivo, bbox_inches='tight')

        if mostrar:
            plt.show()

        return plt

