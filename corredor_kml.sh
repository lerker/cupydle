
#!/bin/bash

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/root/cupydle/

var=""
HOST=$(hostname)
TODAY=$(date)
var="${var}-------------------------------------------------------------------------\n"
var="${var}Fecha: $TODAY                                                  Host:$HOST\n"
var="${var}-------------------------------------------------------------------------\n"


DATA_KML_PCA="all_videos_features_clases_shuffled_PCA85_minmax.npz"
DATA_KML="all_av_features_clases_shuffled_minmax.npz"
UNITin=230300
EPOCAST=50
EPOCASF=300

ARCHIVO_LOG="logK$(date '+_%d%m_%H%M')"
echo -e $var > ${ARCHIVO_LOG}

### ejecutar
##
##  DBN KML PCA
##
###
###

##
unbuffer python3 cupydle/test/dbn_prueba.py \
--dataset ${DATA_KML_PCA} \
--directorio "test_DBN_kml1" \
--capa 85 50 6 \
--epocasTRN ${EPOCAST} \
--epocasFIT ${EPOCASF} \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.01 \
--lrFIT 0.1 \
--gibbs 5 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria"  \
2>&1 | tee -a ${ARCHIVO_LOG}
#
unbuffer python3 cupydle/test/dbn_prueba.py \
--dataset ${DATA_KML_PCA} \
--directorio "test_DBN_kml2" \
--capa 85 50 150 150 50 6 \
--epocasTRN ${EPOCAST} \
--epocasFIT ${EPOCASF} \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.01 \
--lrFIT 0.1 \
--gibbs 5 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria"  \
2>&1 | tee -a ${ARCHIVO_LOG}
#
unbuffer python3 cupydle/test/dbn_prueba.py \
--dataset ${DATA_KML_PCA} \
--directorio "test_DBN_kml3" \
--capa 85 6 \
--epocasTRN ${EPOCAST} \
--epocasFIT ${EPOCASF} \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.01 \
--lrFIT 0.1 \
--gibbs 5 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria"  \
2>&1 | tee -a ${ARCHIVO_LOG}
#
unbuffer python3 cupydle/test/dbn_prueba.py \
--dataset ${DATA_KML_PCA} \
--directorio "test_DBN_kml4" \
--capa 85 500 6 \
--epocasTRN ${EPOCAST} \
--epocasFIT ${EPOCASF} \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.01 \
--lrFIT 0.1 \
--gibbs 5 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria"  \
2>&1 | tee -a ${ARCHIVO_LOG}

##
## KML COMPLETO
unbuffer python3 cupydle/test/dbn_prueba.py \
--dataset ${DATA_KML} \
--directorio "test_DBN_kml5" \
--capa ${UNITin} 1000 50 6 \
--epocasTRN 10 \
--epocasFIT 5 \
--batchsize 10 \
--porcentaje 0.8 \
--lrTRN 0.01 \
--lrFIT 0.1 \
--gibbs 1 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria" \
2>&1 | tee -a ${ARCHIVO_LOG}
##
##
#
echo -e "\n-\n-\n \n \n VALIDACION CRUZADA \n-\n-\n \n \n" >> ${ARCHIVO_LOG}
#
##
##
unbuffer python3 cupydle/test/dbn_prueba_CV.py \
--dataset ${DATA_KML} \
--directorio "test_DBN_kml6" \
--capa ${UNITin} 1000 1000 100 6 \
--epocasTRN 50 \
--epocasFIT 200 \
--batchsize 10 \
--porcentaje 0.08 \
--lrTRN 0.01 \
--lrFIT 0.1 \
--gibbs 3 \
--nombre "dbn" \
--pcd \
--reguL1 0.0 \
--reguL2 0.0 \
--tolErr 0.02 \
--momentoTRN 0.0 \
--momentoFIT 0.0 \
--tipo "binaria" \
--fold 6
2>&1 | tee -a ${ARCHIVO_LOG}


##
##  DBN KML PCA 85
#python3 cupydle/test/dbn_prueba.py --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" --directorio "test_DBN_kml_pca85" --capa 85 50 6 2>&1 | tee -a ${ARCHIVO_LOG}

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
