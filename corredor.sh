
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

ARCHIVO_LOG="log$(date '+_%d%m_%H%M')"
echo -e $var > ${ARCHIVO_LOG}

chmod +x cupydle/test/mnist/get_data.sh

### ejecutar
# mlp comun
#python3 optirun python3 cupydle/test/mlp_FACE.py --directorio "test_MLP" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 50 6 --lepocaTRN 50 -lrTRN 0.01 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn comun
#python3 cupydle/test/dbn_FACE.py --directorio "test_${data}" --dataset ${data} -b 10 --epocasDBN 15 --epocasMLP 50 --capas 85 100 50 25 6 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn que compara un mpl con una dbn entrenadas por separado, puede hacerse ahora haciendo primero un fit y luego un train
#python3 cupydle/test/dbn_FACE_comparativo.py --directorio test_DBN_PCA_85 --dataset $1 -b 10 --epocasDBN 15 --epocasMLP 50 --capas 85 100 50 25 6 -mrd 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn con validacion cruzada
#python3 cupydle/test/dbn_FACE_validacionCruzada.py --directorio "test_DBN" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 50 6 --lepocaTRN 5 --lepocaFIT 10 -lrTRN 0.01 --folds 6 2>&1 | tee -a ${ARCHIVO_LOG}

# mlp sobre los datos puros
#python3 cupydle/test/mlp_FACE.py --directorio "test_MLP2" --dataset "all_av_features_clases_shuffled_minmax.npz" -l 230346  6 --lepocaTRN 500 -lrTRN 0.1 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn sobre los datos puros
#python3 cupydle/test/dbn_FACE.py --directorio "test_DBN" --dataset "all_av_features_clases_shuffled_minmax.npz" -l 230346 500 50 6 --lepocaTRN 50 --lepocaFIT 2000 -lrTRN 0.1 -lrFIT 0.01 --unidadVis binaria --tolErr 0.08 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn sobre los datos PCA 85
#python3 cupydle/test/dbn_FACE.py --directorio "test_DBN3" --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" -l 85 80 70 60 50 40 20 10 6 --lepocaTRN 1000 --lepocaFIT 100000 -lrTRN 0.08 -lrFIT 0.08 --unidadVis binaria --tolErr 0.08 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn grid seach sobre los PCA 85
#python3 cupydle/test/dbn_KML_gridSearch.py --directorio KML --dataset "all_videos_features_clases_shuffled_PCA85_minmax.npz" --capa1 85 60 6 --capa2 85 50 6 --capa3 85 30 6 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn grid search sobre los datos puros
#python3 cupydle/test/dbn_KML_gridSearch.py --directorio KML_video --dataset "all_av_features_clases_shuffled_minmax.npz" --capa1 230346 1000 6 --capa2 230346 500 6 2>&1 | tee -a ${ARCHIVO_LOG}

# dbn grid search sobre MNIST
python3 cupydle/test/dbn_MNIST_gridSearch.py --directorio MNIST --dataset "mnist_minmax.npz" --capa1 784 500 10 --capa2 784 500 500 2000 10 2>&1 | tee -a ${ARCHIVO_LOG}


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
