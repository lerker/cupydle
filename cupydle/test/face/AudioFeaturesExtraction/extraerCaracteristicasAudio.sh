#!/bin/bash

# clases para cada tipo de emocion
# an = Anger
# di = disgust
# fe = Fear
# ha = Happiness
# sa = Sadness
# su = Surprise

an=1
di=2
fe=3
ha=4
sa=5
su=6

cantidad_archivos=`find . -iname "*.wav" | wc -l`
echo "Cantidad de Archivos a Procesar: " $cantidad_archivos

#for AUDIO_FILE in `find . -iname "*video_features.csv"|sort -t'/' -k4`
for AUDIO_FILE in $(find . -iname "*.wav" | awk '{print $NF,$0}' | sort | cut -f2- -d' ')
do
    echo "Procesando: " $AUDIO_FILE
    #aaa=$(basename "$AUDIO_FILE")
    aaa=${AUDIO_FILE##*/}
    TIPO=""
    # el nombre del archivo contine en sus dos primeros caracteres (despues de
    # el backslash) la codificacion del tipo de emocion (dos letras).
    TIPO=$(echo ${aaa} | cut -c1-2)
    CLASE=0

    case $TIPO in
        "an")
            CLASE=$an
            ;;
        "di")
            CLASE=$di
            ;;
        "fe")
            CLASE=$fe
            ;;
        "ha")
            CLASE=$ha
            ;;
        "sa")
            CLASE=$sa
            ;;
        "su")
            CLASE=$su
            ;;
    esac


    matlab -nodesktop -nojvm -nodisplay -nosplash -r "features(${AUDIO_FILE});exit"

done

#matlab -nodesktop -nosplash -r "get_audio_features('~/Pruebas/RML_Emotion_Database_audio/s1/f1/');exit"
