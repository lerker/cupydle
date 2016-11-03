""" Use this file to check the speed difference between a big matrix multiplication
performed on the GPU or on the CPU.
"""
from theano import function, config, shared, tensor, sandbox
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


# para poder ejecutar theano con la gpu es necesario que se encuentre instalado cuda
# cuda 7.5 solo funciona con la version hasta 4.9 de gcc
# la mas actual de gcc es 6.1
# realizar un downgrade de gcc a 4.9 requiere
## quitar las versiones de gcc-libs-multilib y fortran (dependecias de gcc)
#### sudo pacman -Rdd gcc-fortran gcc-libs-multilib
## instalar las versiones de gcc-libs mas vieja
#### downgrade gcc-libs (numero 15)
## por ultimo hacer un downgrade de gcc (numero 15)
#### downgrade gcc

#THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,force_device=True sudo optirun python cupydle/test/checkGPUspeed.py
#https://wiki.archlinux.org/index.php/GPGPU#Using_CUDA_with_an_older_GCC

