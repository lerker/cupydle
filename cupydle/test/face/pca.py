#!/bin/prython
# python pca.py all_videos_features_clases.npz 70 all_videos_pca_70

import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import decomposition

archivo_entrada      = str(sys.argv[1])
cantidad_componentes = int(sys.argv[2])
archivo_salida       = str(sys.argv[3])

print("Archivo entrada: ", archivo_entrada)
print("Archivo salida: ", archivo_salida)
print("Cantidad de componentes: ", cantidad_componentes)

data   = np.load(archivo_entrada)
videos = data['videos']
clases = data['clases']

# las clases estan desde 1..6, deben ser desde 0..5
#clases -= 1
del data #libera memoria

pca = decomposition.PCA(n_components=cantidad_componentes)
pca.fit(videos)
videos = pca.transform(videos)

np.savez_compressed(archivo_salida + '.npz', videos=videos, clases=clases)
