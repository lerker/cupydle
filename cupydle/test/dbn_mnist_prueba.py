#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"


"""
"""

# dependecias internar
import os, argparse, shelve, subprocess, sys, numpy as np

# dependecias propias
from cupydle.test.mnist.mnist import MNIST

directorioActual = os.getcwd()                               # directorio actual de ejecucion
rutaTest    = directorioActual + '/cupydle/test/mnist/'      # sobre el de ejecucion la ruta a los tests
rutaDatos    = directorioActual + '/cupydle/data/DB_mnist/'   # donde se almacenan la base de datos
carpetaTest  = 'test_DBN/'                                  # carpeta a crear para los tests
rutaCompleta    = rutaTest + carpetaTest

if not os.path.exists(rutaCompleta):        # si no existe la crea
    print('Creando la carpeta para el test en: ',rutaCompleta)
    os.makedirs(rutaCompleta)


subprocess.call(rutaTest + 'get_data.sh', shell=True)   # chequeo si necesito descargar los datos


setName = "mnist2"
#MNIST.prepare(rutaDatos, nombre=setName, compresion='bzip2')
data = MNIST(path=rutaDatos)

entrenamiento, validacion, testeo = data.get_all()

print("Digitos Entrenamiento: ", entrenamiento[0].shape, "Etiquetas: ", entrenamiento[1].shape)
print("Digitos Validacion: ", validacion[0].shape, "Etiquetas: ", validacion[1].shape)
print("Digitos Testeo: ", testeo[0].shape, "Etiquetas: ", testeo[1].shape)

data.info


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


#data = entrenamiento[1]
#data = validacion[1]
data = testeo[1]

fig, ax = plt.subplots()

# al usar numpy tenemos mas control
counts, bins = np.histogram(data, bins = 10)

#calculo todos los centros de cada bin
bin_centers = (bins[:-1] + bins[1:])/2

# calculo en ancho que debe tener cada bin
width = 0.7*(bins[1]-bins[0])

# se grafica en diagrama de barras, en los centros, la cantidad, alineacion, ancho, color, borde y patron
plt.bar(bin_centers, counts, align = 'center', width = width, fill=False, edgecolor='black', hatch="...")

# seteo los ticks del eje de las x luego, ahora lo borro
ax.set_xticks([])

# El eje tiene formato decimal
#ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))


# cambiar el color de los bins?
#-
#twentyfifth, seventyfifth = np.percentile(data, [25, 75])
#for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
#    if rightside < twentyfifth:
#        patch.set_facecolor('green')
#    elif leftside > seventyfifth:
#        patch.set_facecolor('red')
#


# Label the raw counts and the percentages below the x-axis...
contador = 0
for count, x in zip(counts, bin_centers):
    # Ticks, labels
    ax.annotate(str(contador) , xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -4), textcoords='offset points', va='top', ha='center')
    contador +=1

    # Label the raw counts
    ax.annotate('%d' % count, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.1f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')



# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.15)

plt.ylabel("Cantidad de ejemplos")

centro_diagrama = (bins[-1] - bins[0]) * 0.5
ax.annotate("Clases", xy=(centro_diagrama, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -46), textcoords='offset points', va='top', ha='center')

#plt.title("Cantidad de ejemplos de entrenamiento por clase")
plt.savefig('cantidad_ejemplos_testeo.png', transparent=True)
plt.savefig('cantidad_ejemplos_testeo.pdf', transparent=True)
plt.show()






##############

data1 = entrenamiento[1]
data2 = validacion[1]
data3 = testeo[1]

fig, ax = plt.subplots()

# al usar numpy tenemos mas control
counts1, bins1 = np.histogram(data1, bins = 10)
counts2, bins2 = np.histogram(data2, bins = 10)
counts3, bins3 = np.histogram(data3, bins = 10)

#calculo todos los centros de cada bin
bin_centers = (bins1[:-1] + bins1[1:])/2

# calculo en ancho que debe tener cada bin
width = 0.7*(bins1[1]-bins1[0])

# se grafica en diagrama de barras, en los centros, la cantidad, alineacion, ancho, color, borde y patron
ax.bar(bin_centers, counts1+counts2+counts3, bottom=bin_centers, align = 'center', width = width, color='w', edgecolor='black', hatch="...", label="Testeo")
ax.bar(bin_centers, counts1+counts2,bottom=bin_centers, align = 'center', width = width, color='w', edgecolor='black', hatch="\\\\", label="Validacion")
ax.bar(bin_centers, counts1, align = 'center', width = width, color='w', edgecolor='black', hatch="///", label="Entrenamiento")
#ax.bar(bin_centers, bottom=counts3, align = 'center', width = width, fill=False, edgecolor='black', hatch="x")

# seteo los ticks del eje de las x luego, ahora lo borro
ax.set_xticks([])

# El eje tiene formato decimal
#ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))





# Label the raw counts and the percentages below the x-axis...
contador = 0
for count, x in zip(counts, bin_centers):
    # Ticks, labels
    ax.annotate(str(contador) , xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -4), textcoords='offset points', va='top', ha='center')
    contador +=1
    """
    # Label the raw counts
    ax.annotate('%d' % count, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.1f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')
    """


# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.15)

plt.ylabel("Cantidad de ejemplos")

centro_diagrama = (bins1[-1] - bins1[0]) * 0.5
ax.annotate("Clases", xy=(centro_diagrama, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -16), textcoords='offset points', va='top', ha='center')

#plt.title("Cantidad de ejemplos de entrenamiento por clase")
plt.savefig('cantidad_ejemplos_totales.png', transparent=True)
plt.savefig('cantidad_ejemplos_totales.pdf', transparent=True)
plt.savefig('cantidad_ejemplos_totales.svg', transparent=True)
plt.savefig('cantidad_ejemplos_totales.eps', transparent=True)

plt.legend(loc=4)
plt.show()



#############################
"""
# guardar
#nombre_archivo_salida='mnist'
#np.savez_compressed(nombre_archivo_salida + '.npz', entrenamiento=entrenamiento[0], entrenamiento_clases=entrenamiento[1], validacion=validacion[0], validacion_clases=validacion[1], testeo=testeo[0], testeo_clases=testeo[1])

def z_score(datos):
    # Normalizacion estadistica (Z-score Normalization)
    # https://en.wikipedia.org/wiki/Standard_score
    # datos [N x D] (N # de datos, D su dimensionalidad).
    # x' = (x - mu)/sigma
    # x = dato de entrada; x' = dato normalizado
    # mu = media; sigma = desvio estandar
    mu = np.mean(datos, axis = 0)
    sigma = np.std(datos, axis = 0) + 0.0005
    return (datos - mu) / sigma

def min_max_escalado(datos, min_obj=0.0, max_obj=1.0):
    # Normalizacion Min-Max
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    # x' = (x-min)/(max-min)*(max_obj-min_obj)+min_obj
    # x = datos de entrada; x' = dato normalizado
    # max_obj = limite superior del rango objetivo
    # min_obj = limite inferior del rango objetivo
    minimo = np.min(datos)
    maximo = np.max(datos)
    x = ((datos - minimo) / (maximo - minimo)) * ( max_obj - min_obj) + min_obj
    return x

entrenamiento = (z_score(entrenamiento[0]), entrenamiento[1])
validacion = (z_score(validacion[0]), validacion[1])
testeo  = (z_score(testeo[0]), testeo[1])

#entrenamiento = (min_max_escalado(z_score(entrenamiento[0])), entrenamiento[1])
#validacion = (min_max_escalado(z_score(validacion[0])), validacion[1])
#testeo  = (min_max_escalado(z_score(testeo[0])), testeo[1])


nombre_archivo_salida='mnist_zscore'
np.savez_compressed(nombre_archivo_salida + '.npz', entrenamiento=entrenamiento[0], entrenamiento_clases=entrenamiento[1], validacion=validacion[0], validacion_clases=validacion[1], testeo=testeo[0], testeo_clases=testeo[1])
"""
images=[]

#images = entrenamiento[0][:10]
indice = 0
while len(images) != 10:
    if entrenamiento[1][indice]==len(images):
        images.append(entrenamiento[0][indice])
    indice+=1

fig = plt.figure()
images = [np.reshape(f, (-1, 28)) for f in images]

#if crop:
#    images = [image[:, 3:25] for image in images]
import matplotlib
image = np.concatenate(images, axis=1)
ax = fig.add_subplot(1, 1, 1)
ax.matshow(image, cmap = matplotlib.cm.binary)
plt.xticks(np.array([]))
plt.yticks(np.array([]))
"""
if save == 'png' or save is True:
    plt.savefig(self.path + "tenDigits" + ".png", format='png')
elif save == 'eps':
    plt.savefig(self.path + 'tenDigits' + '.eps', format='eps', dpi=1000)
elif save == 'svg':
    plt.savefig(self.path + 'tenDigits' + '.svg', format='svg', dpi=1000)
else:
    pass
"""
plt.tight_layout()
plt.savefig("diezDigitos" + '.eps', format='eps', dpi=1000, transparent=True)
plt.savefig("diezDigitos" + '.png', format='png', dpi=1000, transparent=True)
plt.show()

