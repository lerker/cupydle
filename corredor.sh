#!/bin/bash
var=""
HOST=$(hostname)
TODAY=$(date)
var="${var}-------------------------------------------------------------------------\n"
var="${var}Fecha: $TODAY                                                  Host:$HOST\n"
var="${var}-------------------------------------------------------------------------\n"

ARCHIVO_LOG="log$(date '+_%d%m_%H%M')"
echo -e $var > ${ARCHIVO_LOG}

### ejecutar
python3 cupydle/test/dbn_FACE.py --directorio test_DBN_PCA_85 --dataset videos_pca_85_features_procesado_minmax.npz -b 10 --epocasDBN 15 --epocasMLP 50 --capas 85 100 50 25 6 -m 2>&1 | tee -a ${ARCHIVO_LOG}


FECHA_ARCHIVO_FIN=$(date)
var2=""
var2="${var2}\n-------------------------------------------------------------------------\n"
var2="${var2}FINALIZADO Fecha: ${FECHA_ARCHIVO_FIN} \n"
var2="${var2}-------------------------------------------------------------------------\n"
echo -e $var2 >> ${ARCHIVO_LOG}

# con el comando tee se puede direccionar la salida a un archivo
# ls -lR 2>&1 | tee coso.salida
#2>&1
#y agregando stdbuf -o 0 al inicio hace que no se guarde nada en el buffer
# stdbuf -o 0 python3
