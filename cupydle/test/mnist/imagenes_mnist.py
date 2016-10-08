
# dependecias propias



import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os, subprocess, numpy as np


def crear():
    # correr en el root de cupydle
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

    #MNIST.prepare(rutaDatos, nombre=setName, compresion='bzip2')
    data = MNIST(path=rutaDatos)

    entrenamiento, validacion, testeo = data.get_all()

    print("Digitos Entrenamiento: ", entrenamiento[0].shape, "Etiquetas: ", entrenamiento[1].shape)
    print("Digitos Validacion: ", validacion[0].shape, "Etiquetas: ", validacion[1].shape)
    print("Digitos Testeo: ", testeo[0].shape, "Etiquetas: ", testeo[1].shape)

    data.info

    return entrenamiento, validacion, testeo

def guardar(entrenamiento, validacion , testeo):
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

    return 1

def histo(data, hatch, tam=13):
    #data = entrenamiento[1]
    #data = validacion[1]
    #data = testeo[1]

    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)

    # al usar numpy tenemos mas control
    counts, bins = np.histogram(data, bins = 10)

    #calculo todos los centros de cada bin
    bin_centers = (bins[:-1] + bins[1:])/2

    # calculo en ancho que debe tener cada bin
    width = 0.7*(bins[1]-bins[0])

    # se grafica en diagrama de barras, en los centros, la cantidad, alineacion, ancho, color, borde y patron
    plt.bar(bin_centers, counts, align = 'center', width = width, fill=False, edgecolor='black', hatch=hatch)

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
            xytext=(0, -4), textcoords='offset points', va='top', ha='center', size=tam)
        contador +=1

        # Label the raw counts
        ax.annotate('%d' % count, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points', va='top', ha='center', size=tam)

        # Label the percentages
        percent = '%0.1f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -32), textcoords='offset points', va='top', ha='center', size=tam)



    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)

    plt.ylabel("Cantidad de ejemplos", size=tam+2)

    centro_diagrama = (bins[-1] - bins[0]) * 0.5
    ax.annotate("Clases", xy=(centro_diagrama, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -46), textcoords='offset points', va='top', ha='center', size=tam+2)

    #plt.title("Cantidad de ejemplos de entrenamiento por clase")
    fig.tight_layout()
    plt.savefig('histograma_data.png', bbox_inches='tight' ,transparent=True)
    plt.savefig('histograma_data.pdf', bbox_inches='tight' ,transparent=True)
    plt.show()

    return 0


def histo_todos(data1, data2, data3, tam=13):
    #data1 = entrenamiento[1]
    #data2 = validacion[1]
    #data3 = testeo[1]

    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)

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
    for count, x in zip(counts1, bin_centers):
        # Ticks, labels
        ax.annotate(str(contador) , xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -4), textcoords='offset points', va='top', ha='center', size=tam)
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

    plt.ylabel("Cantidad de ejemplos", size=tam+2)

    centro_diagrama = (bins1[-1] - bins1[0]) * 0.5
    ax.annotate("Clases", xy=(centro_diagrama, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -16), textcoords='offset points', va='top', ha='center', size=tam+2)

    #plt.title("Cantidad de ejemplos de entrenamiento por clase")
    fig.tight_layout()
    plt.savefig('cantidad_ejemplos_totales.png', bbox_inches='tight' ,transparent=True)
    plt.savefig('cantidad_ejemplos_totales.pdf', bbox_inches='tight', transparent=True)
    plt.savefig('cantidad_ejemplos_totales.svg', bbox_inches='tight', transparent=True)
    plt.savefig('cantidad_ejemplos_totales.eps', bbox_inches='tight', transparent=True)

    plt.legend(loc=4)

    plt.show()
    return 0

def plot_10digits(imagenes_muestra):
    plt.gray()
    fig = plt.figure( figsize=(16,7) )
    for i in range(0,10):
        ax = fig.add_subplot(1,10,i+1)
        ax.matshow(imagenes_muestra[i].reshape((28,28)).astype(float), cmap='gray', interpolation='nearest')
    plt.show()

def plot_30random(imagenes):
    plt.gray()
    fig = plt.figure( figsize=(16,7) )
    for i in range(0,30):
        ax = fig.add_subplot(3,10,i+1)
        ax.matshow(imagenes[i].reshape((28,28)).astype(float))
    plt.show()


def medias(imagenes):
    means = {i: [] for i in range(10)}
    for i in range(10):
        #mask = labels == i
        #means[i] = imagenes[mask]
        means[i] = np.mean(imagenes[labels == i], axis=0)
    return means

def mnist_10means(means):
    plt.gray()
    fig = plt.figure( figsize=(16,7) )
    for i in range(0,10):
        ax = fig.add_subplot(1,10,i+1)
        ax.matshow(means[i].reshape((28,28)).astype(float), cmap='gray', interpolation='nearest')
    fig.tight_layout()
    plt.savefig('Mnist_10medias.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Mnist_10medias.png', bbox_inches='tight', dpi=300)
    plt.show()
    return 0

def mnist_medias(imagenes_muestra, means):
    plt.gray()
    fig = plt.figure( figsize=(10,9) )
    for i in range(0,20):
        ax = fig.add_subplot(4,5,i+1)
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        if i >= 0 and i < 5:
            ax.matshow(imagenes_muestra[i].reshape((28,28)).astype(float))
            ax.set_xlabel("Digito: " + str(i), fontsize=14)
        elif i >= 5 and i < 10:
            ax.matshow(means[i-5].reshape((28,28)).astype(float))
            ax.set_xlabel("Digito media: " + str((i-5)), fontsize=13)
        elif i >= 10 and i < 15:
            ax.matshow(imagenes_muestra[i-5].reshape((28,28)).astype(float))
            ax.set_xlabel("Digito: " + str((i-5)), fontsize=14)
        elif i >= 15 and i < 20:
            ax.matshow(means[i-10].reshape((28,28)).astype(float))
            ax.set_xlabel("Digito media: " + str((i-10)), fontsize=13)

    fig.tight_layout()
    plt.savefig('Mnist_medias.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Mnist_medias.png', bbox_inches='tight', dpi=300)
    plt.show()
    return 0


def plot_mejor_iteracion():
    import numpy as np
    """
    cat iteracion | cut -d ' ' -f12 - > iteracion_error_validacion

    cat iteracion | cut -d ' ' -f8 - > iteracion_costo
    cat iteracion_costo | tr -d "," > iteracion_costo.csv
    """
    entrenamiento = np.loadtxt('iteracion_costo.csv')
    validacion = np.loadtxt('iteracion_error_validacion')
    import matplotlib.pyplot as plt
    ax1=plt.subplot(211)
    plt.semilogx(entrenamiento, linewidth=1.0, color='r')
    ax1.set_ylabel('log(costo)')
    plt.title('Costo Entrenamiento')


    ax2=plt.subplot(212)
    plt.semilogx(validacion,linewidth=1.0)
    ax2.set_ylabel('log(error)')
    plt.title('Error de validación')

    plt.tight_layout()
    plt.savefig('ErroresMejorMNIST.png', bbox_inches='tight' ,transparent=True)
    plt.savefig('ErroresMejorMNIST.pdf', bbox_inches='tight' ,transparent=True)
    plt.show()

if __name__ == "__main__":

    data=np.load('mnist_minmax.npz')

    X_x = data['entrenamiento']
    X_c = data['entrenamiento_clases']

    Y_x = data['validacion']
    Y_c = data['validacion_clases']

    Z_x = data['testeo']
    Z_c = data['testeo_clases']

    imagenes = np.concatenate((X_x, Y_x, Z_x), axis=0)
    labels = np.concatenate((X_c, Y_c, Z_c), axis=0)

    imagenes_muestra=np.zeros((10,784))

    indice = 0; contador = 0
    while contador != 10:
        if labels[indice]==contador:
            imagenes_muestra[contador] = imagenes[indice]
            contador+=1
        indice+=1


    #means=medias(imagenes)

    #mnist_medias(imagenes_muestra, means)

    histo(Z_c, hatch='....')
    #histo_todos(X_c, Y_c, Z_c)
