#!/usr/bin/env python3
#    mutual_information_parallel is a function for estimating Mutual Information, in parallel
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

import numpy as np
import multiprocessing as mp

def pyMIestimator(X,Y,k=5,base=np.exp(1)):

    N = X.shape[0] #  The number of samples
    
    p = mp.Pool(processes=mp.cpu_count())
    
    # Since all the counts, nx and ny, are added after calculating 
    # their psi (digamma function), we can parallelise them by 
    # calculating the sum of psi of nx and ny at each ith instance
    # Each worker/process is passed with X, Y, k and i as single
    # list argument, makes it super easy to implement
    sum_psi = p.map(get_sum_psi_nxi_nyi,
                     [[X, Y, k, i] for i in range(N)])
    sum_psi = sum(sum_psi)

    # Estimating Mutual Information:
    I2 = (digamma(k) - 1/float(k)
          - (sum_psi / N)
          + digamma(N) ) / np.log(base)

    # No negative MI:
    if I2 < 0: I2 = 0

    return I2
    
def get_sum_psi_nxi_nyi(args):
    ''' Function to get sum of psi of nx and ny at ith instance
    Takes is one list comprising X, Y, K and index i
    Passing one arg makes it super easy to use Pool'''
    X, Y, k, i = args
    
    # Matrices to buffer i'th instance of X & Y, N times
    buffer_X = np.zeros(X.shape)
    buffer_Y = np.zeros(Y.shape)
    
    buffer_X[:] = X[i]
    dx = np.absolute(buffer_X - X)
    for j in reversed(range(1, len(X.shape))):
        dx = np.linalg.norm(dx, axis=j)
    dxS = np.delete(dx,i,0)
    # Scaling
    buffer_max = np.amax(dxS)
    dxS_scaled = dxS / (buffer_max if buffer_max != 0 else 1)

    buffer_Y[:] = Y[i]
    dy = np.absolute(buffer_Y - Y)
    for j in reversed(range(1, len(Y.shape))):
        dy = np.linalg.norm(dy, axis=j)
    dyS = np.delete(dy, i, 0)
    # Scaling
    buffer_max = np.amax(dyS)
    dyS_scaled = dyS / (buffer_max if buffer_max != 0 else 1)

    # Z space
    dzS = np.maximum(dxS_scaled, dyS_scaled)

    dzSort = np.argsort(dzS) #indices of sorted array
    
    # Epsn is epsilon_n(i)/2
    # Epsn is the distance from sample n(i) to its k-th neighbor
    # in z space
    # where n=x,y
    Epsx = dxS_scaled[dzSort[k-1]]
    Epsy = dyS_scaled[dzSort[k-1]]
    # we replace nx(i) and ny(i) by the number of points with
    # ||x(i)-x(j)|| <= Epsx(i) and ||y(i)-y(j)|| <= Epsy(i)
    nx2 = sum(dxS_scaled <= Epsx)
    ny2 = sum(dyS_scaled <= Epsy)
    
    sum_psi = digamma(nx2) + digamma(ny2)
    return sum_psi

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
    
