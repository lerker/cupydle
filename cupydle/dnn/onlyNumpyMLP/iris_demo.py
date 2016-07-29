

# Librerias de Python
import time
import os

# Librerias internas


print("Cargando base de datos ...")
# data = datasets.load_iris()
# features = data.data
# labels = data.target
dataset = LabeledDataSet()
dataset.load_file(os.curdir+'/datasets/iris.csv')
print("Size de la data: ")
print(dataset.shape)

print("Creando conjuntos de train, valid y test ...")
train, valid, test = dataset.split_data([.7, .2, .1])  # Particiono conjuntos
# Standarize data
std = StandardScaler()