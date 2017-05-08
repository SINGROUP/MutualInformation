__author__ = 'binzam1'
import numpy as np
from pyMIestimator import pyMIestimator

# test case:
k=5
X=np.linspace(0,10,num=100)
X=np.transpose(X)
[N,]=X.shape
X=np.reshape(X, (N,1))
Y=np.sin(X)

#plt.figure(2)
#plt.plot(X, Y, 'bo')  # plo
#plt.grid()
#plt.show()

k=5

[I1,I2]=pyMIestimator(X,Y,k)
print I1, I2

