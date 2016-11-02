#!/bin/bash

# Script que genera un ARCHIVO con la concatenacion de todos los features
# que ya fueron extraidos de los videos y se encuentran almacenados en sus
# respectivos directorios.
# Cada línea corresponde a la concatenación de frames de un video.

# Version modificada
# Forma de uso:
# $ ./concatenarVideos.sh all_videos.csv

# Ponzoni Nelson, npcuadra<at>gmail.com

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

if [ $# -eq 0 ]; then
    echo "Debe ingresar el nombre del archivo .csv a guardar"
    exit
fi

#ARCHIVO=all_videos.csv
ARCHIVO=$1
if [ -e  $ARCHIVO ]; then
    echo "Cuidado el archivo ya existe!!"
    echo "Desea agregar al final del mismo? (s/n)"

    read -n 1 respuesta

    if [ "$respuesta" != "s" ]; then
        exit 1
    fi
fi

cantidad_archivos=`find . -iname "*video_features.csv" | wc -l`
echo "Cantidad de Archivos a Procesar: " $cantidad_archivos

#for VIDEO_FILE in `find . -iname "*video_features.csv"|sort -t'/' -k4`
for VIDEO_FILE in $(find . -iname "*video_features.csv" | awk '{print $NF,$0}' | sort | cut -f2- -d' ')
do
    echo "Procesando: " $VIDEO_FILE
    #aaa=$(basename "$VIDEO_FILE")
    aaa=${VIDEO_FILE##*/}
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

    #if [ $TIPO = "an" ]; then
    #    CLASE=$an

    # buffer para acumular todos los frames de un video
    ACUM=""
    while read LINE
    do
       ACUM="${ACUM} ${LINE}" #concatena en el buffer
    done < ${VIDEO_FILE}

    # concatena al inicio del buffer la clase
    ACUM="${CLASE} ${ACUM}"

    # guarda a disco el buffer (concatena al archivo)
	echo ${ACUM} >> ${ARCHIVO}
done

## shuffling el archivo?
#shuf all_videos_features_clases.csv -o all_videos_features_clases_shuffled.csv
#shuf ${ARCHIVO} -o "${ARCHIVO}_shuffled.csv"
