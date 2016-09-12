import numpy as np
from numpy import genfromtxt
import sys

# se le pasa por argumentos el nombre del archivo
# 'all_videos_features_clases.csv'
nombre_archivo=sys.argv[0]

videos = genfromtxt(nombre_archivo, delimiter=' ', dtype=np.float32)

data   = videos[:,1:].astype(np.float32)
clases = videos[:,0].astype(np.int8)

print("Archivo:\t",       nombre_archivo)
print("Archivo crudo:\t", videos.shape)
print("Videos crudo:\t",  data.shape,   "\ttipo: ", data.dtype)
print("Clases:\t\t",      clases.shape, "\ttipo: ", clases.dtype)


# Assume input data matrix X of size [N x D]
def normalizacion(data):
    # http://cs231n.github.io/neural-networks-2/
    # media cero y desvio unitario
    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    return data

data = normalizacion(data)

###
"""
from six.moves import cPickle
f = open('obj.save', 'wb')
cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
"""

"""
np.save('videos_normalizados.npy', data)
np.save('clases.npy', clases)

nueva = np.load("videos_normalizados.npy")

print(nueva.shape)
print(nueva[0:0:100])
"""

np.savez_compressed('videos_clases.npz', videos=data, clases=clases)
#b = np.load('videos_clases.npz')
#print(b['videos'])
#print(b['clases'])




