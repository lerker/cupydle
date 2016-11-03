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
Estructura basica para la implementacion basica de una DBN sobre CUALQUIER
conjunto de datos!!!
"""

# dependecias internar
import os, argparse, shelve, sys, numpy as np

# dependecias propias
from cupydle.dnn.utils import temporizador
from cupydle.dnn.dbn import DBN
from cupydle.dnn.gridSearch import ParameterGrid
from cupydle.dnn.validacion_cruzada import train_test_split

parametros = {  'general':          ['mnist'],
                'directorio':       ['dbn_dir'],
                'dataset':          [None],
                'nombre':           ['dbn'],
                'tipo':             ['binaria','gaussiana'],
                'capas':            [],
                'epocasTRN':        [[3]],
                'epocasFIT':        [3],
                'tambatch':         [10],
                'tasaAprenTRN':     [0.1],
                'tasaAprenFIT':     [0.1],
                'regularizadorL1':  [0.0],
                'regularizadorL2':  [0.0],
                'momentoTRN':       [0.0],
                'momentoFIT':       [0.0],
                'pasosGibbs':       [1],
                'porcentaje':       [0.8],
                'toleranciaError':  [0.1],
                'pcd':              [True, False]}

class dbn_base_test(object):

    def __init__(self, **kwargs):

        # re-parseo de los parametros pasados
        self.general         = kwargs['general']
        self.directorio      = kwargs['directorio']
        self.dataset         = kwargs['dataset']
        self.nombre          = kwargs['nombre']
        self.tipo            = kwargs['tipo']
        self.capas           = kwargs['capas']
        self.epocasTRN       = kwargs['epocasTRN']
        self.epocasFIT       = kwargs['epocasFIT']
        self.tambatch        = kwargs['tambatch']
        self.tasaAprenTRN    = kwargs['tasaAprenTRN']
        self.tasaAprenFIT    = kwargs['tasaAprenFIT']
        self.regularizadorL1 = kwargs['regularizadorL1']
        self.regularizadorL2 = kwargs['regularizadorL2']
        self.momentoTRN      = kwargs['momentoTRN']
        self.momentoFIT      = kwargs['momentoFIT']
        self.pasosGibbs      = kwargs['pasosGibbs']
        self.porcentaje      = kwargs['porcentaje']
        self.toleranciaError = kwargs['toleranciaError']
        self.pcd             = kwargs['pcd']

        self.capas           = np.asarray(self.capas)
        self.tasaAprenTRN    = np.asarray([self.tasaAprenTRN]) if isinstance(self.tasaAprenTRN, float) else np.asarray(self.tasaAprenTRN)
        self.momentoTRN      = np.asarray([self.momentoTRN]) if isinstance(self.momentoTRN, float) else np.asarray(self.momentoTRN)
        self.pasosGibbs      = np.asarray([self.pasosGibbs]) if isinstance(self.pasosGibbs, int) else np.asarray(self.pasosGibbs)

        # chequeos
        assert self.dataset.find('.npz') != -1, "El conjunto de datos debe ser del tipo '.npz'"
        assert len(self.epocasTRN) >= len(self.capas) or len(self.epocasTRN) == 1, "Epocas de entrenamiento y cantidad de capas no coinciden (unidad aplica a todas)"
        assert len(self.tasaAprenTRN) >= len(self.capas) or len(self.tasaAprenTRN) == 1, "Tasa de aprendizaje no coincide con la cantidad de capas (unidad aplica a todas)"
        assert len(self.momentoTRN) >= len(self.capas) or len(self.momentoTRN) == 1, "Tasa de momento entrenamiento no coincide con la cantidad de capas (unidad aplica a todas)"
        assert len(self.pasosGibbs) >= len(self.capas) or len(self.pasosGibbs) == 1, "Pasos de Gibbs no coinciden con la cantidad de capas (unidad aplica a todas)"
        assert self.porcentaje <= 1.0

        # ajustes
        self.epocasTRN = self.epocasTRN * len(self.capas) if len(self.epocasTRN) == 1 else self.epocasTRN
        self.tasaAprenTRN = np.resize(self.tasaAprenTRN, (len(self.capas),)) if len(self.tasaAprenTRN) == 1 else self.tasaAprenTRN
        self.momentoTRN = np.resize(self.momentoTRN, (len(self.capas),)) if len(self.momentoTRN) == 1 else self.momentoTRN
        self.pasosGibbs = np.resize(self.pasosGibbs, (len(self.capas),)) if len(self.pasosGibbs) == 1 else self.pasosGibbs


        # configuraciones con respecto a los directorios
        self.directorioActual = os.getcwd() # directorio actual de ejecucion
        self.rutaTest         = self.directorioActual + '/cupydle/test/'+ self.general+'/'  # sobre el de ejecucion la ruta a los tests
        self.rutaDatos        = self.directorioActual + '/cupydle/data/DB_'+ self.general+'/' # donde se almacenan la base de datos
        self.carpetaTest      = self.directorio + '/'                   # carpeta a crear para los tests
        self.rutaCompleta     = self.rutaTest + self.carpetaTest
        os.makedirs(self.rutaCompleta) if not os.path.exists(self.rutaCompleta) else None # crea la carpeta en tal caso que no exista

        # se crea el modelo
        self.miDBN = DBN(nombre=self.nombre, ruta=self.rutaCompleta)
        DBN.DBN_custom = True

        # se agregan las capas
        for idx in range(len(self.capas[:-1])): # es -2 porque no debo tener en cuenta la primera ni la ultima
            self.miDBN.addLayer(n_visible    = self.capas[idx],
                           n_hidden     = self.capas[idx+1],
                           epocas       = self.epocasTRN[idx],
                           tamMiniBatch = self.tambatch,
                           lr_pesos     = self.tasaAprenTRN[idx],
                           pasosGibbs   = self.pasosGibbs[idx],
                           w            = None,
                           momento      = self.momentoTRN[idx],
                           tipo         = self.tipo)
        return

    def entrenar(self, data):
        ###########################################################################
        ##
        ##          P R E - E N T R E N A M I E N T O        R B M s
        ##
        ###########################################################################
        datosDBN = data
        cant_clases = len(np.bincount(datosDBN[0][1]))
        print("                    Clases                     :", "".join("{:^6}".format(x) for x in range(0,cant_clases)))
        print("                                               :", "".join("{:^6}".format("-----") for x in range(0,cant_clases)))
        print("Cantidad de clases en el conjunto EntrenamieDBN:", "".join("{:^6}".format(x) for x in np.bincount(datosDBN[0][1])))
        print("Entrenado la DBN con {} ejemplos".format(len(datosDBN[0][1])))

        #entrena la red
        tiempo_entrenar = self.miDBN.entrenar( dataTrn=datosDBN[0][0], # imagenes de entrenamiento
                                               dataVal=None, # imagenes de validacion
                                               pcd=self.pcd,
                                               guardarPesosIniciales=False,
                                               filtros=True)

        #miDBN.save(rutaCompleta + "dbnMNIST", compression='zip')
        del datosDBN
        return tiempo_entrenar

    def ajustar(self, datos):
        ###########################################################################
        ##
        ##                 A J U S T E     F I N O    ( M L P )
        ##
        ###########################################################################
        datosMLP=datos
        cant_clases = len(np.bincount(datosMLP[0][1]))
        print("                    Clases                     :", "".join("{:^6}".format(x) for x in range(0,cant_clases)))
        print("                                               :", "".join("{:^6}".format("-----") for x in range(0,cant_clases)))
        print("Cantidad de clases en el conjunto Entrenamiento:", "".join("{:^6}".format(x) for x in np.bincount(datosMLP[0][1])))
        print("Cantidad de clases en el conjunto Validacion: \t", "".join("{:^6}".format(x) for x in np.bincount(datosMLP[1][1])))
        print("Cantidad de clases en el conjunto Test: \t", "".join("{:^6}".format(x) for x in np.bincount(datosMLP[2][1])))

        #miDBN = DBN.load(filename=rutaCompleta + "dbnMNIST", compression='zip')
        print(self.miDBN)

        parametros={'tasaAprendizaje':  self.tasaAprenFIT,
                    'regularizadorL1':  self.regularizadorL1,
                    'regularizadorL2':  self.regularizadorL2,
                    'momento':          self.momentoFIT,
                    'toleranciaError':  self.toleranciaError,
                    'epocas':           self.epocasFIT}
        self.miDBN.setParametros(parametros)

        costoTRN, costoVAL, costoTST, costoTST_final, tiempo_ajustar = self.miDBN.ajuste( datos=datosMLP,
                                                                     listaPesos=None,
                                                                     fnActivacion="sigmoidea",
                                                                     semillaRandom=None,
                                                                     tambatch=self.tambatch)

        del datosMLP

        #miDBN.guardarObjeto(nombreArchivo=nombre)

        return costoTRN, costoVAL, costoTST, costoTST_final, tiempo_ajustar
        # FIN DEL AJUSTE FINO

def _guardar(nombreArchivo, valor):

    nombreArchivo = nombreArchivo + '.cupydle'

    with shelve.open(nombreArchivo, flag='c', writeback=False, protocol=2) as shelf:
            for key in valor.keys():
                shelf[key] = valor[key]
            shelf.close()

    return 0

def test():
    assert len(sys.argv) == 2, "cantidad incorrecta de parametros"
    parametros['dataset'] = [sys.argv[1]]
    parametros['capas'] = [[784, 100, 10]]
    #parametros['dataset'] = ["mnist_minmax.npz"]
    #parametros['capas'] = [[784, 100, 10], [85, 30, 6]]

    Grid = ParameterGrid(parametros)
    cantidad_combinaciones = len(Grid)
    cantidad_a_ejecutar = cantidad_combinaciones // 2

    nombreArchivo = 'resultados_dbnMNIST_gridSearch'
    print("GUARDANDO LOS RESULTADOS EN EL ARCHIVO {} QUE CONTIENE {} ITERACIONES\n\n".format(nombreArchivo, cantidad_a_ejecutar))

    T = temporizador()
    inicio = T.tic()

    for x in range(cantidad_a_ejecutar):
        T2 = temporizador()
        inicio2 = T2.tic()

        print("****************************************************")
        print("\n\n")
        print("Iteracion {} de {}".format(x, cantidad_a_ejecutar))
        print("PARAMETROS:")
        # modo random la eleccion de los parametros sobre el conjunto posible
        indice = np.random.randint(cantidad_combinaciones)
        # no tengo implementada el __setitem__ en ParameterGrid
        params = Grid[indice]
        params['directorio'] = 'dbn_grid_' + str(x)
        for k in sorted(params.keys()):
            print(str("{: >25} : {: <50}").format(k, str(params[k])))
        print("\n\n")
        d = dbn_base_test(**params)

        ##
        ###########################################################################
        ##
        ##   P R E P A R A N D O   L O S   D A T O S   E N T R E N A M I E N T O
        ##
        ###########################################################################

        # se cargan  los datos, debe ser un archivo comprimido, en el cual los
        # arreglos estan en dos keys, 'videos' y 'clases'
        directorioActual = os.getcwd() # directorio actual de ejecucion
        rutaDatos = str(directorioActual) + '/cupydle/data/DB_' + str(parametros['general'][0]) +'/' # donde se almacenan la base de datos
        try:
            datos = np.load(rutaDatos + str(sys.argv[1]))
        except:
            assert False, "El dataset no existe en la ruta: " + rutaDatos + str(sys.argv[1])

        entrenamiento        = datos['entrenamiento'].astype(np.float32)
        entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
        del datos # libera memoria

        datosDBN = []; datosDBN.append((entrenamiento, entrenamiento_clases))
        del entrenamiento, entrenamiento_clases
        ##
        d.entrenar(data=datosDBN)
        del datosDBN


        ##
        ###########################################################################
        ##
        ##   P R E P A R A N D O   L O S   D A T O S   A J U S T E  F I N O
        ##
        ###########################################################################

        # se cargan  los datos, debe ser un archivo comprimido, en el cual los
        # arreglos estan en dos keys, 'videos' y 'clases'
        try:
            datos = np.load(rutaDatos + str(sys.argv[1]))
        except:
            assert False, "El dataset no existe en la ruta: " + rutaDatos + str(sys.argv[1])

        entrenamiento        = datos['entrenamiento'].astype(np.float32)
        entrenamiento_clases = datos['entrenamiento_clases'].astype(np.int32)
        validacion           = datos['validacion'].astype(np.float32)
        validacion_clases    = datos['validacion_clases'].astype(np.int32)
        testeo               = datos['testeo'].astype(np.float32)
        testeo_clases        = datos['testeo_clases'].astype(np.int32)
        del datos # libera memoria

        datosMLP = []
        datosMLP.append((entrenamiento, entrenamiento_clases))
        datosMLP.append((validacion, validacion_clases))
        datosMLP.append((testeo, testeo_clases))
        del entrenamiento, entrenamiento_clases, validacion, validacion_clases
        del testeo, testeo_clases
        ##
        costoTRN, costoVAL, costoTST, costoTST_final = d.ajustar(datos=datosMLP)
        del datosMLP
        #

        _guardar(nombreArchivo=nombreArchivo, valor={str(x): {'parametros':params, 'costoTRN':costoTRN, 'costoVAL':costoVAL, 'costoTST':costoTST, 'costoTST_final':costoTST_final }})

        final2 = T2.toc()
        print("\n\nPaso de la grilla terminado, iteracion " + str(x+1) + "\n\n")
        print("Tiempo: {}".format(T2.transcurrido(inicio2, final2)))
        print("****************************************************")
        print("\n\n")

    final = T.toc()
    print("\n\nGRID SEARCH  FINALIZADO\n\n")
    print("Tiempo total requerido: {}".format(T.transcurrido(inicio, final)))


    return 0

if __name__ == '__main__':
    test()




