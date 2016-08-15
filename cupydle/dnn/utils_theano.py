
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
