from MutualInformation import pyMIestimator
import sys
sys.path.append('/u/58/reischt1/unix/ml_projects/readAFM/src')
import readAFMHDF5 as rHDF5


db = rHDF5.AFMdata('/l/reischt1/toyDB_v14_twoAtoms3D.hdf5', [41,41,41,1])

k = 5

for sz in range(10, 30, 5):
    for amp in [7.0, 10.0, 15.0]:
        data = db.batch_runtimeSolution(200, sigmabasez = float(sz), amplificationFactor = amp, verbose = False)
        X = data['forces']
        Y = data['solutions']
        
        I=pyMIestimator(X,Y,k)
        print('I(sz= '+str(sz)+'; amp= '+str(amp)+') = '+str(I))
        with open('Mutual_information-toyDB_v14-k'+str(k)+'.dat','a') as f:
            f.write('I(sz= '+str(sz)+'; amp= '+str(amp)+') = '+str(I)+'\n')