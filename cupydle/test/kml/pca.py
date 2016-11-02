#!/bin/prython
# python pca.py all_videos_features_clases.npz 70 all_videos_pca_70

import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import decomposition

archivo_entrada      = str(sys.argv[1])
cantidad_componentes = int(sys.argv[2])
archivo_salida       = str(sys.argv[3])


print("Archivo entrada: ",  archivo_entrada)
print("Archivo salida: ",   archivo_salida + '.npz')
print("Cantidad de componentes: ", cantidad_componentes)

### funcion de ayuda que carga un archivo gigande csv
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
###

videos = None
clases = None
if archivo_entrada.find('.npz') != -1: # es un archivo comprimido, por lo que supongo que debo estraer las cosas...
    data   = np.load(archivo_entrada)
    videos = data['videos']
    clases = data['clases']
    del data #libera memoria
elif archivo_entrada.find('.csv') != -1: # es un tipo csv crudo, cargo con la ayuda
    data   = iter_loadtxt(filename=archivo_entrada, delimiter=' ', dtype=np.float32, skiprows=0)
    videos = data[:,1:].astype(np.float32)
    clases = data[:,0].astype(np.int8)
    del data #libera memoria
else:
    assert False, "ARCHIVO NO PROCESABLE con la terminacion dada"

print("realizando PCA sobre el archivo")
pca = decomposition.PCA(n_components=cantidad_componentes)
pca.fit(videos)
videos = pca.transform(videos)
print("FIN")

np.savez_compressed(archivo_salida + '.npz', videos=videos, clases=clases)
