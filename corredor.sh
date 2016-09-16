#!/bin/bash


optirun cupydle/test/dbn_FACE.py --directorio test_DBN_PCA_85 --dataset videos_pca_85_features_procesado_minmax.npz -b 10 --epocasDBN 15 --epocasMLP 500 --capas 85 100 50 25 6 -mrd
