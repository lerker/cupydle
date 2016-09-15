#!/bin/bash


# Forma de uso:
# $ ./contatenarAudio.sh

# Ponzoni Nelson, npcuadra<at>gmail.com

# dicho script busca todos los archivos de audio y video de la carpeta y
# concatena en uno solo, al cual la primer columna le corresponde la clase
# seguida por n columnas correspondientes al video y luego por m columnas
# correspondientes al archivo de audio. Un ejemplo por fila


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

cantidad_archivos=`find . -iname "*-audio_cropRaw.csv" | wc -l`
echo "Cantidad de Archivos a Procesar: " $cantidad_archivos


#for VIDEO_FILE in `find . -iname "*video_features.csv"|sort -t'/' -k4`
for AUDIO_FILE in $(find . -iname "*-audio_cropRaw.csv" | awk '{print $NF,$0}' | sort | cut -f2- -d' ')
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

    #if [ $TIPO = "an" ]; then
    #    CLASE=$an

    # buffer para acumular todos los frames de un video
    ACUM=""

    # el archivo es de una sola linea por lo que no tiene sentido que recorra
    # todo pero lo dejo por las dudsa por si despues lo cambiamos
    while read LINE
    do
        ACUM="${ACUM} ${LINE}" #concatena en el buffer
    done < ${AUDIO_FILE}

    # concatena al inicio del buffer la clase # ver lo del espacio aca
    ACUM="${CLASE} ${ACUM}"

    # guarda a disco el buffer (concatena al archivo)
    echo ${ACUM} >> ${ARCHIVO}
done

## shuffling el archivo?
#shuf all_videos_features_clases.csv -o all_videos_features_clases_shuffled.csv
#shuf ${ARCHIVO} -o "${ARCHIVO}_shuffled.csv"
