# Dependencias Internas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Dependencias Externas
from cupydle.dnn.NeuralNetwork import NeuralNetwork
from cupydle.dnn.data import LabeledDataSet as dataSet


# Seteo de los parametros
capa_entrada = 2
capa_oculta_1 = 3
capa_salida = 1
capas = [capa_entrada, capa_oculta_1, capa_salida]

net = NeuralNetwork(list_layers=capas)


# --------------------------------  DATASET  --------------------------------------------#
print("Cargando la base de datos...")
data_trn = dataSet('cupydle/data/XOR_trn.csv', delimiter=',')
print("Entrenamiento: " + str(data_trn.data.shape))
entrada_trn, salida_trn = data_trn.split_data(3)

# corregir los valores de los labels, deben estar entre 0 y 1 "clases"
for l in range(0, len(salida_trn)):
    if salida_trn[l] < 0.0:
        salida_trn[l] = 0.0
    else:
        salida_trn[l] = 1.0

datos_trn = [(x, y) for x, y in zip(entrada_trn, salida_trn)]

data_tst = dataSet('cupydle/data/XOR_tst.csv', delimiter=',')
print("Testeo: " + str(data_tst.data.shape))
entrada_tst, salida_tst = data_tst.split_data(3)

# corregir los valores de los labels, deben estar entre 0 y 1 "clases"
for l in range(0, len(salida_tst)):
    if salida_tst[l] < 0.0:
        salida_tst[l] = 0.0
    else:
        salida_tst[l] = 1.0

datos_tst = [(x, y) for x, y in zip(entrada_tst, salida_tst)]

# --------------------------------  PLOTS  --------------------------------------------#

fig = plt.figure()
plt.clf()
plt1 = fig.add_subplot(2, 1, 1, projection='3d')
plt2 = fig.add_subplot(2, 1, 2, projection='3d')


scat1_x, scat1_y, scat1_z = [], [], []
scat2_x, scat2_y, scat2_z = [], [], []

for l in range(0, len(entrada_trn)):
    if salida_trn[l] == 1.0:
        scat1_x.append(entrada_trn[l, 0])
        scat1_y.append(entrada_trn[l, 1])
        scat1_z.append(salida_trn[l])
    else:
        scat2_x.append(entrada_trn[l, 0])
        scat2_y.append(entrada_trn[l, 1])
        scat2_z.append(salida_trn[l])

c1 = ['r'] * len(scat1_z)
c2 = ['b'] * len(scat2_z)

plt1.scatter(xs=scat1_x, ys=scat1_y, zs=scat1_z, c=c1, s=5.0, marker='+')
plt1.scatter(xs=scat2_x, ys=scat2_y, zs=scat2_z, c=c2, s=5.0, marker='o')

plt1.set_xlabel('X')
plt1.set_ylabel('Y')
plt1.set_zlabel('Label')
plt1.set_title("Conjunto de Entrenamiento")

scat1_x, scat1_y, scat1_z = [], [], []
scat2_x, scat2_y, scat2_z = [], [], []

for l in range(0, len(entrada_tst)):
    if salida_tst[l] == 1.0:
        scat1_x.append(entrada_tst[l, 0])
        scat1_y.append(entrada_tst[l, 1])
        scat1_z.append(salida_tst[l])
    else:
        scat2_x.append(entrada_tst[l, 0])
        scat2_y.append(entrada_tst[l, 1])
        scat2_z.append(salida_tst[l])

c1 = ['r'] * len(scat1_z)
c2 = ['b'] * len(scat2_z)

plt2.scatter(xs=scat1_x, ys=scat1_y, zs=scat1_z, c=c1, s=5.0, marker='+')
plt2.scatter(xs=scat2_x, ys=scat2_y, zs=scat2_z, c=c2, s=5.0, marker='o')

plt2.set_xlabel('X')
plt2.set_ylabel('Y')
plt2.set_zlabel('Label')
plt2.set_title("Conjunto de Testeo")
plt.show()

# --------------------------------  ENTRENAMIENTO --------------------------------------------#

tmp = datos_trn
cantidad = len(tmp)
porcentaje = 80
datos_trn = tmp[0: int(cantidad * porcentaje / 100)]
datos_vld = tmp[int(cantidad * porcentaje / 100 + 1):]
print("Training Data size: " + str(len(datos_trn)))
print("Validating Data size: " + str(len(datos_vld)))
print("Testing Data size: " + str(len(datos_tst)))


net.fit(train=datos_trn, valid=datos_vld, test=datos_tst, batch_size=1, epocas=20, tasa_apren=0.2, momentum=0.1)
