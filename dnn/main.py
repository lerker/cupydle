#!/usr/bin/env python3

from numpy import genfromtxt
import numpy as np

## leo los datos
data = genfromtxt('irisbin.csv', delimiter=',')
print("http://www.python-course.eu/numpy.php")

c = np.array([[4, 3], [2, 1]])
d = np.array([[1, 2], [3, 4]])

# imprimo por pantalla
#print(c)
#print(d)

# multiplicacion matricial
#print("mul matrix\n", np.dot(c,d))
# [[13 20]
#  [ 5  8]]

# multiplicacion elemento a elemento
#print("mul elemt\n", c*d)

# TODO http://matlabgeeks.com/tips-tutorials/neural-networks-a-multilayer-perceptron-in-matlab/
#creo un conjunto de datos de prueba
# XOR input for x1 and x2
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# salida del XOR
output = np.array([[0, 1], [1, 0]])

# inicializacion del bias
bias = np.array([[-1], [-1], [-1]])

print("INPUT\n",input)
print("OUTPUT\n",output)
print("BIAS\n",bias)


#
#
#
# Learning coefficient
coeff = 0.7
# Number of learning iterations
iterations = 10000
# Calculate weights randomly using seed.
#Random values in a given shape.
#Create an array of the given shape and propagate it with random samples from a uniform distribution over [0, 1).
np.random.seed(seed=10) # fijo la semilla
weights = -0.5 + np.random.rand(3,3)

print(weights)

##
##
##
# TRAIN
from train import train
train([3,2,1],input,output,10,1,1,1)
"""
for i = 1:iterations
   out = zeros(4,1);
   numIn = length (input(:,1));
   for j = 1:numIn
      % Hidden layer
      H1 = bias(1,1)*weights(1,1)
          + input(j,1)*weights(1,2)
          + input(j,2)*weights(1,3);

      % Send data through sigmoid function 1/1+e^-x
      % Note that sigma is a different m file
      % that I created to run this operation
      x2(1) = sigma(H1);
      H2 = bias(1,2)*weights(2,1)
           + input(j,1)*weights(2,2)
           + input(j,2)*weights(2,3);
      x2(2) = sigma(H2);

      % Output layer
      x3_1 = bias(1,3)*weights(3,1)
             + x2(1)*weights(3,2)
             + x2(2)*weights(3,3);
      out(j) = sigma(x3_1);

      % Adjust delta values of weights
      % For output layer:
      % delta(wi) = xi*delta,
      % delta = (1-actual output)*(desired output - actual output)
      delta3_1 = out(j)*(1-out(j))*(output(j)-out(j));

      % Propagate the delta backwards into hidden layers
      delta2_1 = x2(1)*(1-x2(1))*weights(3,2)*delta3_1;
      delta2_2 = x2(2)*(1-x2(2))*weights(3,3)*delta3_1;

      % Add weight changes to original weights
      % And use the new weights to repeat process.
      % delta weight = coeff*x*delta
      for k = 1:3
         if k == 1 % Bias cases
            weights(1,k) = weights(1,k) + coeff*bias(1,1)*delta2_1;
            weights(2,k) = weights(2,k) + coeff*bias(1,2)*delta2_2;
            weights(3,k) = weights(3,k) + coeff*bias(1,3)*delta3_1;
         else % When k=2 or 3 input cases to neurons
            weights(1,k) = weights(1,k) + coeff*input(j,1)*delta2_1;
            weights(2,k) = weights(2,k) + coeff*input(j,2)*delta2_2;
            weights(3,k) = weights(3,k) + coeff*x2(k-1)*delta3_1;
         end
      end
   end
end
"""