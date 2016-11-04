import theano, numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theanoFloat = theano.config.floatX

class coso(object):

    def __init__(self):
        self.buf   = theano.shared(value=numpy.ones(shape=(2,3), dtype=theanoFloat), name='buf')
        self.buf2   = theano.shared(value=numpy.zeros(shape=(1,), dtype=theanoFloat), name='buf2')
        self.buf3   = theano.shared(value=numpy.zeros(shape=(2,3), dtype=theanoFloat), name='buf3')
        self.gen = RandomStreams(seed=1000)
        self.inn = theano.shared(value=numpy.asarray([1,1.5], dtype=theanoFloat), name='entrada')
    def actualizar(self, val):

        alfa = theano.tensor.cast(val,  dtype=theanoFloat)
        suma = theano.tensor.sum(self.buf)
        maxx = theano.tensor.max(self.buf)

        valorRand = numpy.random.randn(1).astype(theanoFloat)
        #prob = self.buf/maxx
        prob = self.buf + numpy.asarray([[1.0, 0.5, 0.3], [0.01, 0.6, 0.9]], dtype=theanoFloat)
        valorRand2 = self.gen.binomial(size=prob.shape, n=1, p=prob, dtype=theanoFloat)
        updates = [(self.buf, valorRand2), (self.buf3, prob)]
        return updates

    def eje(self):
        updates = self.actualizar(2.0)

        fun = theano.function(inputs=[],outputs=[self.buf, self.buf3], updates=updates)

        print(fun())
        print(fun())
        print(fun())
        print(fun())
        print(fun())
        return 0

if __name__ == "__main__":

    a = coso()

    a.eje()


