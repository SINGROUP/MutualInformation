import numpy as np

def pyMIestimator(X,Y,k=5):
# pyMIestimator is a function for estimating Mutual Information
#
# Inputs function:
# X,Y         : an N x M matrix (samples x features)
# k           : the number of nearest neighbour
#
# Outputs function:
# I1, I2      : mutual information estimates
#
# Martha Arbayani Zaidan, PhD
# Postdoctoral Research Fellow
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

    [N,M] = X.shape #  The number of samples

    # Calculate the distance between each data point (sample)
    # and its k-th nearest neighbour:
    dx = np.zeros((N,N))
    dy = np.zeros((N,N))
    dz = np.zeros((N,N))

    for i in range(0,N):
        for j in range(0,N):
            dx[i,j] = np.sqrt(np.sum((X[i,:] - X[j,:])**2))
            dy[i,j] = np.sqrt(sum((Y[i,:] - Y[j,:])**2))
            dz[i,j] = max([dx[i,j], dy[i,j]])


    # Count nx(i) and ny(i) using the criteria defined in Kraskov et al. (2004)
    Eps = np.zeros((N,1))
    nx1 = np.zeros((N,1))
    ny1 = np.zeros((N,1))
    nx2 = np.zeros((N,1))
    ny2 = np.zeros((N,1))

    for i in range(0,N):

    # the first cols of {dx,dy,dz} are zeros because the first cols are
    # the distance between themselves, so we remove them here:
        dxS = dx[i,:]
        dxS[i] = np.nan
        idx=np.logical_not(np.isnan(dxS))
        dxS=dxS[idx]

        dyS = dy[i,:]
        dyS[i] = np.nan
        idx=np.logical_not(np.isnan(dyS))
        dyS=dyS[idx]

        dzS = dz[i,:]
        dzS[i] = np.nan
        idx=np.logical_not(np.isnan(dzS))
        dzS=dzS[idx]

        dzSort = np.sort(dzS)
        # Eps(i) is epsilon(i)/2
        # Eps(i) is the distance from sample z(i) to its k-th neighbor
        Eps[i] = dzSort[k-1]

        # For algorithm 1:
        # count the number nx(i) of points x(j) whose distance from x(i) is
        # strictly less than Eps, and similarly for y instead of x:
        nx1[i] = sum(dxS < Eps[i])
        ny1[i] = sum(dyS < Eps[i])

        # For algorithm 2:
        # we replace nx(i) and ny(i) by the number of points with
        # ||x(i)-x(j)|| <= Eps(i) and ||y(i)-y(j)|| <= Eps(i)
        nx2[i] = sum(dxS <= Eps[i])
        ny2[i] = sum(dyS <= Eps[i])

    # Estimating Mutual Information:
    I1 = digamma(k) - (sum(digamma(nx1 + 1) + digamma(ny1 + 1)) / N) + digamma(N)
    I2 = digamma(k) - 1.0/k - (sum(digamma(nx2) + digamma(ny2)) / N) + digamma(N)

    I1=np.sign(I1)*np.sqrt(1-np.exp(-2*abs(I1)))
    I2=np.sign(I2)*np.sqrt(1-np.exp(-2*abs(I2)))

    # No negative MI:
    if I1 < 0: I1 = 0
    elif I2 < 0: I2 = 0

    return I1, I2

def digamma ( x ):
# digamma function calculates digamma(x) = d(log(gamma(x)))/dX
# Authors:
#
#    Original FORTRAN77 version by Jose Bernardo.
#    Original Python version by John Burkardt.
#    Modified Python version by Martha Arbayani Zaidan.
#    This modified version takes numpy array input.
#
#  Reference:
#
#    Jose Bernardo,
#    Algorithm AS 103:
#    Psi ( Digamma ) Function,
#    Applied Statistics,
#    Volume 25, Number 3, 1976, pages 315-317.

  x=np.float64(x) # convert to be double precision

  #  Check the input.
  if ( x.any <= 0.0 ):
    value = 0.0
    return value

  #  Initialize.
  value = 0.0

  #  Use approximation for small argument.

  if ( x.any <= 0.000001 ):
    # Euler-Mascheroni constant:
    euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
    value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
    return value

  #  Reduce to DIGAMA(X + N).

  while ( x.any < 8.5 ):
    value = value - 1.0 / x
    x = x + 1.0

#  Use Stirling's (actually de Moivre's) expansion.

  r = 1.0 / x
  value = value + np.log ( x ) - 0.5 * r
  r = r * r
  value = value \
    - r * ( 1.0 / 12.0
    - r * ( 1.0 / 120.0
    - r * ( 1.0 / 252.0
    - r * ( 1.0 / 240.0
    - r * ( 1.0 / 132.0 ) ) ) ) )

  return value
