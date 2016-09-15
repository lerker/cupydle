
__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"


import theano
import theano.misc.pkl_utils
import theano.sandbox.cuda.basic_ops as sbcuda
import theano.tensor
import numpy


def load(filename=None, compression='gzip'):
  objeto = None
  with open(filename,'rb') as f:
    objeto = theano.misc.pkl_utils.load(f)
    f.close()
  return objeto
  # END LOAD

def save(objeto, filename=None, compression='gzip'):
  with open(filename + '.zip','wb') as f:
    # arreglar esto
    """
        def params(self):
        parametros = [v for k, v in self.__dict__.items()]
        print(parametros)

        return parametros
    """
    import theano.misc.pkl_utils
    theano.misc.pkl_utils.dump(objeto.params(), f)
    f.close()
  return

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, theano.tensor.cast(shared_y, 'int32')

def gpu_info(conversion='Gb'):
    """
    Retorna la informacion relevante de la GPU (memoria)
    Valores REALES, en base 1024 y no 1000 como lo expresa el fabricante

    :type conversion: str
    :param conversion: resultado devuelto en Mb o Gb (default)
    """
    # es una tupla de la forma
    # (cantidad de momoria libre, cantidad de momeria total)
    memoriaLibreBytes, memoriaTotalBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()

    # memoria fisica total
    memoriaTotal = memoriaTotalBytes/1024./1024.
    memoriaTotal = (memoriaTotal/1024.) if conversion == 'Gb' else memoriaTotal

    # memoria disponible
    memoriaLibre = memoriaLibreBytes/1024./1024.
    memoriaLibre = (memoriaLibre/1024.) if conversion == 'Gb' else memoriaLibre

    memoriaOcupada = memoriaTotal - memoriaLibre
    porcentajeMemOcu = (memoriaOcupada/memoriaTotal) * 100.

    return memoriaLibre, memoriaOcupada, memoriaTotal, porcentajeMemOcu

def calcular_chunk(memoriaDatos, tamMiniBatch, cantidadEjemplos, porcentajeUtil=0.9):
    """
    calcula el tamMacroBatch
    cuantos ejemplos deben enviarse a la gpu por pedazos.

    """
    tamMacroBatch = None
    contador = 1
    memoriaGPU = gpu_info('Mb')[0] * porcentajeUtil
    print("MEMORIA GPU",memoriaGPU)
    while True:
        a = memoriaGPU / (memoriaDatos / contador)
        b = (cantidadEjemplos / contador) % tamMiniBatch == 0

        if a >= 1.0 and b:
            break

        contador +=1
        if contador == 10000000:
            print("A",a,"B",b)
            assert False, "contador de MacroBatch demasiado alto " + str(contador)

    if contador == 1:
        tamMacroBatch = cantidadEjemplos
    else:
        tamMacroBatch = contador * tamMiniBatch

    return tamMacroBatch

