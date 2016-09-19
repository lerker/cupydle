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
# mlp comun
#python3 optirun python3 cupydle/test/mlp_FACE.py --directorio "test_MLP" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 50 6 --lepocaTRN 50 -lrTRN 0.01 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn comun
#python3 cupydle/test/dbn_FACE.py --directorio "test_${data}" --dataset ${data} -b 10 --epocasDBN 15 --epocasMLP 50 --capas 85 100 50 25 6 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn que compara un mpl con una dbn entrenadas por separado, puede hacerse ahora haciendo primero un fit y luego un train
#python3 cupydle/test/dbn_FACE_comparativo.py --directorio test_DBN_PCA_85 --dataset $1 -b 10 --epocasDBN 15 --epocasMLP 50 --capas 85 100 50 25 6 -mrd 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn con validacion cruzada
python3 cupydle/test/dbn_FACE_validacionCruzada.py --directorio "test_DBN" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 50 6 --lepocaTRN 5 --lepocaFIT 10 -lrTRN 0.01 --folds 6 2>&1 | tee -a ${ARCHIVO_LOG}

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
