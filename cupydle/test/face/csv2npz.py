#!/bin/prython3
# python3 preprocesamiento.py all_videos_features_clases.npz "coso_procesado" 'z_score+min_max'

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
archivo_entrada = sys.argv[1]

# archivo de salida comprimido
nombre_archivo_salida = sys.argv[2]

# como tratar los datos
tipo = str(sys.argv[3])
tipo = np.float32 if "float"==tipo else np.int32

videos = None
clases = None

data   = iter_loadtxt(filename=archivo_entrada, delimiter=' ', dtype=tipo, skiprows=0)
videos = data[:,1:].astype(np.float32)
clases = data[:,0].astype(np.int8)
del data #libera memoria

print("Archivo:\t",       archivo_entrada)
print("Archivo crudo:\t", videos.shape+1)
print("Videos crudo:\t",  videos.shape,   "\ttipo: ", videos.dtype)
print("Clases:\t\t",      clases.shape, "\ttipo: ", clases.dtype)


# almacenamiento comprimido
# en un solo archivo, dentro se encuentra cada array con el nombre otorgado en le kword
np.savez_compressed(nombre_archivo_salida + '.npz', videos=videos, clases=clases)
