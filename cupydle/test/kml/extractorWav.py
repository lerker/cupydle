#!/bin/python3

# recorta el audio de una pista .wav en referencia a los frames de inicio y fin
# del video original.
# la salida puede ser elegida si se requiere el archivo .wav o diretamente los
# volcados a un archivo .csv en una sola linea.

#>>> python3 extractorWav.py an1.wav 58 112

import sys
import os
from scipy.io.wavfile import read as scread
#from scipy.io.wavfile import write as scwrite
from numpy import savetxt as npsavetxt
from numpy import resize as npresize

if len(sys.argv) != 4:
    assert False, "cantindad incorrecta de parametros"

nombreArchivo = sys.argv[1]
#Vinicio, Vfin = 58, 112
Vinicio = int(sys.argv[2])  # frame de inicio en el video
Vfin = int(sys.argv[3])     # frame de fin en el video
Vrate = 30.                 # tasa de muestreo de frames en el video

# quito la terminacion para manipular el nombre
nombreArchivo = os.path.splitext(nombreArchivo)[0]
print("Procesando el archivo: ", nombreArchivo + '.wav')

Arate, data = scread(nombreArchivo + '.wav')

ainicio     = int( (Vinicio/Vrate) * Arate )
afin        = int( (Vfin/Vrate) * Arate )
audioCrop   = data[ainicio:afin]

nombreArchivoCrop = nombreArchivo+'-audio_cropRaw'

# en necesario guardar el recortado como wav?
#scwrite(nombreArchivoCrop+'.wav', Arate, audioCrop)

# formato entero signado
npsavetxt(fname=nombreArchivoCrop+'.csv', X=npresize(audioCrop, (1,-1)), delimiter=' ', fmt='%d', newline='\n')
