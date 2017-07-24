#!/usr/bin/env python3
import numpy as np

def pyMIestimator(X,Y,k=5,base=np.exp(1)):
# pyMIestimator is a function for estimating Mutual Information
#
# Inputs function:
# X,Y         : an N x ... tensors (samples N features)
# k           : the number of nearest neighbour
# unit base   : base=2 -> Shannon (bits), base=exp(1)=nat
#
# Outputs function:
# I1, I2      : mutual information estimates
#
# Martha Arbayani Zaidan, PhD
# Postdoctoral Research Fellow
# Aalto University, Finland
#
# Yashasvi Singh Ranawat
# Aalto University, Finland
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

    # Count nx(i) and ny(i) using the criteria defined in Kraskov et al. (2004)
    #nx1 = np.zeros(N)
    #ny1 = np.zeros(N)
    nx2 = np.zeros(N)
    ny2 = np.zeros(N)
    # Matrices to buffer i'th instance of X & Y, N times
    buffer_X=np.zeros(X.shape)
    buffer_Y=np.zeros(Y.shape)

    for i in range(N):
        #Fancy progress bar
        if True and N != 1:
            print('\r'+'({0:-<10})'.format('>'*int(i/float(N-1)*10))+('\n' if (i+1)==N else ''),end='')

        buffer_X[:]=X[i]
        dx=np.absolute(buffer_X-X)
        for j in reversed(range(1,len(X.shape))):
            dx=np.linalg.norm(dx,axis=j)
        dxS = np.delete(dx,i,0)
        #Scaling
        buffer_max=np.amax(dxS)
        dxS_scaled=dxS/(buffer_max if buffer_max != 0 else 1)

        buffer_Y[:]=Y[i]
        dy=np.absolute(buffer_Y-Y)
        for j in reversed(range(1,len(Y.shape))):
            dy=np.linalg.norm(dy,axis=j)
        dyS = np.delete(dy,i,0)
        #Scaling
        buffer_max=np.amax(dyS)
        dyS_scaled=dyS/(buffer_max if buffer_max != 0 else 1)

        #Z space
        dzS=np.maximum(dxS_scaled,dyS_scaled)

        dzSort = np.argsort(dzS) #indices of sorted array
        '''
        ## For algorithm 1:
        # Eps is epsilon(i)/2
        # Eps is the distance from sample z(i) to its k-th neighbor
        Eps = dzS[dzSort[k-1]]
        # count the number nx(i) of points x(j) whose distance from x(i) is
        # strictly less than Eps, and similarly for y :
        nx1[i] = sum(dxS_scaled <= Eps)
        ny1[i] = sum(dyS_scaled <= Eps)
        '''
        ## For algorithm 2:
        # Epsn is epsilon_n(i)/2
        # Epsn is the distance from sample n(i) to its k-th neighbor in z space
        # where n=x,y
        Epsx = dxS_scaled[dzSort[k-1]]
        Epsy = dyS_scaled[dzSort[k-1]]
        # we replace nx(i) and ny(i) by the number of points with
        # ||x(i)-x(j)|| <= Epsx(i) and ||y(i)-y(j)|| <= Epsy(i)
        nx2[i] = sum(dxS_scaled <= Epsx)
        ny2[i] = sum(dyS_scaled <= Epsy)

    # Estimating Mutual Information:
    #I1 = (digamma(k) - ((sum(map(digamma,nx1 + 1)) + sum(map(digamma,ny1 + 1))) / N) + digamma(N) ) / np.log(base)
    I2 = (digamma(k) - 1/float(k) - (sum(list(map(digamma,nx2)) + list(map(digamma,ny2))) / N) + digamma(N) ) / np.log(base)

    # No negative MI:
#    if I1 < 0: I1 = 0
    if I2 < 0: I2 = 0

#    return I1, I2
    return I2

'''
#digamma function using recursion; 
#not used since maximum recursion depth exceeds in comparison
def digamma(x):
    if x==1:
        return -0.57721566490153286060651209008240243104215933593992 #Euler-Mascheroni constant
    else:
        return 1/float(x-1) + digamma(x-1)
'''

#digamma function using while; 
def digamma(x):
    count=1
    value=-0.57721566490153286060651209008240243104215933593992 #Euler-Mascheroni constant
    while count<int(x):
        value+=1/float(count)
        count+=1
    return value
    
##Execution

if __name__ == "__main__":
    import sys,os
    if len(sys.argv)<3:
        print ("\n//Usage: ",sys.argv[0]," <numpy array file 1> <numpy array file 2> <optional: index for .dat file>\n")
        sys.exit()
        
    arg=sys.argv[1:]
    k=5
    for i in range(len(arg)):
        if arg[i].startswith('-k'):
            k=int(arg[i][2:])
            arg.pop(i)
            break

    for i in arg[:1]:
        if not os.path.isfile(i):
            print ("\n//Error: ",i," not found\n")
            sys.exit()
            
    X=np.load(arg[0])
    Y=np.load(arg[1])
    
    if not X.shape[0]==Y.shape[0]:
        print('Array index 1 size mismatch')
        sys.exit()
    
    I=pyMIestimator(X,Y,k)
    with open('Mutual_information-k'+str(k)+'.dat','a') as f:
        f.write('I('+arg[0]+';'+arg[1]+') = '+str(I)+(' '+arg[2] if len(arg)>2 else '')+'\n')
    
