#!/bin/bash

# Forma de uso:
# $ ./extractorAudioRaw.sh

# Ponzoni Nelson, npcuadra<at>gmail.com


cantidad_archivos=`find . -iname "*.wav" | wc -l`
echo "Cantidad de Archivos a Procesar: " $cantidad_archivos

for AUDIO_FILE in $(find . -iname "*.wav" | awk '{print $NF,$0}' | sort | cut -f2- -d' ')
do
    #python3 ./extractorWav.py $AUDIO_FILE
    SIN_EXT=${AUDIO_FILE%%.wav} # a partir del final, quita toda coincidencia con .wav
    SIN_INI=${SIN_EXT##./} # a partit del inicio quita coincidencia y retorna eso
    #echo $SIN_INI
    NOMBRE_ARCHIVO_FRAMES="${SIN_INI}-voiced_frames.txt"
    NOMBRE_ARCHIVO_FRAMES2="${SIN_EXT}-voiced_frames.txt"
    #echo $NOMBRE_ARCHIVO_FRAMES
    FRAMES_INI=$(head -n 1 "${NOMBRE_ARCHIVO_FRAMES2}" | cut -d, -f1)
    FRAMES_FIN=$(head -n 1 "${NOMBRE_ARCHIVO_FRAMES2}" | cut -d, -f2)
    python3 ./extractorWav.py $AUDIO_FILE $FRAMES_INI $FRAMES_FIN
    echo "Frame de inicio: " $FRAMES_INI "Frame de fin: " $FRAMES_FIN
done
