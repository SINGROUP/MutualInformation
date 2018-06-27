#!/usr/bin/env python3
import numpy as np
from scipy.spatial.distance import squareform, pdist

def pyMIestimator(X,Y,k=5,base=np.exp(1)):
#    pyMIestimator is a function for estimating Mutual Information
#    Copyright (C) 2017 Aalto University (Surfaces and Interfaces at 
#    the Nanoscale (SIN)), Finland
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public License
#    as published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this program.  If not, see
#    <http://www.gnu.org/licenses/>.
#
#    Martha Arbayani Zaidan, PhD
#    Postdoctoral Research Fellow
#    Aalto University, Finland
#
#    Yashasvi Singh Ranawat 
#    yashasvi.ranawat@gmail.com
#    Aalto University, Finland
#
# Inputs function:
# X,Y         : an N x ... tensors (samples N features)
# k           : the number of nearest neighbour
# unit base   : base=2 -> Shannon (bits), base=exp(1)=nat
#
# Outputs function:
# I1, I2      : mutual information estimates
#
# References:
#
#    Alexander Kraskov Harald Stogbauer, and Peter Grassberger,
#    Estimating mutual information,
#    Physical review E 69, no. 6 (2004): 066138.
#
#    Jorge Numata, Oliver Ebenhoh and Ernst-Walter Knapp,
#    Measuring correlations in metabolomic networks with mutual information,
#    Genome Informatics 20 (2008): 112-122.

    N = X.shape[0] #  The number of samples
    
    # distance matrices
    dist_X = squareform(pdist(X.reshape((N, -1))))
    dist_Y = squareform(pdist(Y.reshape((N, -1))))

    # scaling
    dist_X /= np.std(dist_X, axis=1, keepdims=True)
    dist_Y /= np.std(dist_Y, axis=1, keepdims=True)

    # index matrix
    dist_Z = np.maximum(dist_X, dist_Y)
    indx = np.argsort(dist_Z)[:, k-1]

    # psi
    psi = 0

    for i in range(N):
        nx = np.sum(dist_X[i, :] <= dist_X[i, indx[i]])
        ny = np.sum(dist_Y[i, :] <= dist_Y[i, indx[i]])
        psi += digamma(nx) + digamma(ny)

    # mutual information
    I = (digamma(k) - 1/float(k)
          - (psi / N)
          + digamma(N) ) / np.log(base)

    # No negative MI:
    if I < 0: I = 0

    return I

'''
#digamma function using recursion; 
#not used since maximum recursion depth exceeds in comparison
def digamma(x):
    if x == 1:
        #Euler-Mascheroni constant
        return -0.57721566490153286060651209008240243104215933593992 
    else:
        return 1/float(x-1) + digamma(x-1)
'''

#digamma function using while; 
def digamma(x):
    count = 1
    #Euler-Mascheroni constant
    value = -0.57721566490153286060651209008240243104215933593992 
    while count < int(x):
        value += 1/float(count)
        count += 1
    return value
    
##Execution

if __name__ == "__main__":
    
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("\n//Usage: ", sys.argv[0],
               " <numpy array file 1> <numpy array file 2>" \
               " <optional: index for .dat file>\n")
        sys.exit()
        
    arg = sys.argv[1:]
    k = 5
    for i in range(len(arg)):
        if arg[i].startswith('-k'):
            k = int(arg[i][2:])
            arg.pop(i)
            break

    for i in arg[:1]:
        if not os.path.isfile(i):
            print("\n//Error: ", i, " not found\n")
            sys.exit()
            
    X = np.load(arg[0])
    Y = np.load(arg[1])
    
    if not X.shape[0] == Y.shape[0]:
        print('Array index 1 size mismatch')
        sys.exit()
    
    I = pyMIestimator(X,Y,k)
    with open('Mutual_information-k' + str(k) + '.dat', 'a') as f:
        f.write('I('+ arg[0] + ';' + arg[1] + ') = ' + str(I)
                + (' ' + arg[2] if len(arg) > 2 else '') + '\n')
    
