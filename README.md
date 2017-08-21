Mutual Information
==================
This provides Mutual Information (mi) functions in Python. Currently, this provides the mi between tensors as described by [Kraskov et.al.](https://arxiv.org/pdf/cond-mat/0305641.pdf)


mutual_information.py
---------------------

It takes in two numpy array files as command line inputs and outputs the mutual information in Mutual_information-k*.dat. The numpy array files should have same number of instances, i.e. same first index size. Those instances can be a tensor.

Usage: 

```python
import mutual_information as mi
import numpy as np

X = np.load('<numpy array file 1>') # X.size = (N,...)
Y = np.load('<numpy array file 2>') # Y.size = (N,...)
I = mi.pymiestimator(X, Y) #default k = 5, base = np.exp(1)
```

```
./mutual_information.py [-k<k value>] <numpy array file 1> <numpy array file 2> [<index for .dat file>]
```

Options:

    -k<k value>
        Number of nearest neighbours to account for, default=5
        
    <index for .dat file>
        number or index for easy plotting against Mutual information from .dat file

Useful properties:
-----------------

Theoretical maximum
Maximum mutual information can be understood to be of a variable with itself, i.e. I(X,X). This is given by:
```python
I = (mi.digamma(N) - mi.digamma(k+1)) / np.log(base) #default k = 5, base = np.exp(1)
```
This also implies that the number of instances cannot be less than or equal to k+1. Good rule of thumb is to have instances >> k (atleast 10 times).

Transfering Mutual Information
To transfer mutual information calculated using two arrays with N instances to M instances:
```python
I_M = I_N + (- mi.digamma(N) + mi.digamma(M)) / np.log(base) #defaule base = np.exp(1)
```
This is useful when comparing mutual information calculated with different number of instances.
