#!/bin/prython3
# python3 preprocesamiento.py all_videos_features_clases.csv "coso_procesado" 'z_score+min_max'

import numpy as np
import sys

def iter_loadtxt(filename, delimiter=' ', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data



# se le pasa por argumentos el nombre del archivo
# 'all_videos_features_clases.csv'
nombre_archivo = sys.argv[1]

# archivo de salida comprimido
nombre_archivo_salida = sys.argv[2]

# tipo de procesamiento: 'z_score', 'min_max', 'blanqueo', 'z_score+min_max'
procesamiento = sys.argv[3]

videos = iter_loadtxt(filename=nombre_archivo, delimiter=' ', dtype=np.float32, skiprows=0)

data   = videos[:,1:].astype(np.float32)
clases = videos[:,0].astype(np.int8)

print("Archivo:\t",       nombre_archivo)
print("Archivo crudo:\t", videos.shape)
print("Videos crudo:\t",  data.shape,   "\ttipo: ", data.dtype)
print("Clases:\t\t",      clases.shape, "\ttipo: ", clases.dtype)


"""
FUNCIONES DE AYUDA
"""
def z_score(datos):
    # Normalizacion estadistica (Z-score Normalization)
    # https://en.wikipedia.org/wiki/Standard_score
    # datos [N x D] (N # de datos, D su dimensionalidad).
    # x' = (x - mu)/sigma
    # x = dato de entrada; x' = dato normalizado
    # mu = media; sigma = desvio estandar
    mu = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    return (datos - mu) / sigma
def z_score(datos):
    # Normalizacion estadistica (Z-score Normalization)
    # https://en.wikipedia.org/wiki/Standard_score
    # datos [N x D] (N # de datos, D su dimensionalidad).
    # x' = (x - mu)/sigma
    # x = dato de entrada; x' = dato normalizado
    # mu = media; sigma = desvio estandar
    mu = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    return (datos - mu) / sigma

def min_max_escalado(datos, min_obj=0.0, max_obj=1.0):
    # Normalizacion Min-Max
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    # x' = (x-min)/(max-min)*(max_obj-min_obj)+min_obj
    # x = datos de entrada; x' = dato normalizado
    # max_obj = limite superior del rango objetivo
    # min_obj = limite inferior del rango objetivo
    minimo = np.min(datos)
    maximo = np.max(datos)
    x = ((datos - minimo) / (maximo - minimo)) * ( max_obj - min_obj) + min_obj
    return x

def blanqueo(datos, eigenvalues=100, epison=1e-5):
    # whitening PCA normalization
    # https://en.wikipedia.org/wiki/Whitening_transformation
    # http://cs231n.github.io/neural-networks-2/
    # Assume input data matrix X of size [N x D]
    datos -= np.mean(datos, axis = 0) # zero-center the data (important)
    cov = np.dot(datos.T, datos) / datos.shape[0] # get the data covariance matrix
    U,S,V = np.linalg.svd(cov)
    datos_rot = np.dot(datos, U) # decorrelate the data
    # similar a PCA...
    # se quedan con los eigenvalues (dimensiones) mas importantes (contienen mas varianza)
    datos_rot_reduced = np.dot(datos, U[:,:eigenvalues]) # Xrot_reduced becomes [N x 100]
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    datos_white = datos_rot / np.sqrt(S + epison)
    return datos_white



# procesando los datos segun el tipo seleccionado
if procesamiento == 'min_max':
    data = min_max_escalado(data)
elif procesamiento == 'z_score':
    data = z_score(data)
elif procesamiento == 'blanqueo':
    data = blanqueo(data)
elif procesamiento == 'z_score+min_max':
    data = min_max_escalado(z_score(data))
else:
    assert False, "no existe ese tipo de procesamiento"

# almacenamiento comprimido
# en un solo archivo, dentro se encuentra cada array con el nombre otorgado en le kword
np.savez_compressed(nombre_archivo_salida + '.npz', videos=data, clases=clases)




