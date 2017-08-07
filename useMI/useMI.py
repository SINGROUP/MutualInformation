'''
Created on Jul 25, 2017

@author: reischt1
'''

import sys
sys.path.append('/u/58/reischt1/unix/ml_projects/MutualInformation/src')
from MutualInformation import pyMIestimator
sys.path.append('/u/58/reischt1/unix/ml_projects/readAFM/src')
import readAFMHDF5 as rHDF5

k = 5
n = 100

db = rHDF5.AFMdata('/l/reischt1/toyDB_v14_twoAtoms3D.hdf5', [41,41,41,1])



# for sz in range(2, 150):
for sz in [200.]:
    for amp in [10.0]:
        try:
            data = db.batch_runtimeSolution(n, sigmabasez = float(sz)/10., amplificationFactor = amp, verbose = True)
        except KeyError:
            print('KeyError occured. sz = {}, amp = {}'.format(sz, amp))
            continue
        X = data['forces']
        Y = data['solutions']
        print("sz: %f, amp: %f"%(float(sz)/10., amp))
        I=pyMIestimator(X,Y,k)
        print('I(sz= '+str(float(sz)/10.)+'; amp= '+str(amp)+') = '+str(I))
        with open('Mutual_information-toyDB_v14-k'+str(k)+'_amp'+str(amp)+'_n'+str(n)+'.dat','a') as f:
            f.write('sz= '+str(float(sz)/10.)+' amp= '+str(amp)+ ' I= '+str(I)+'\n')
