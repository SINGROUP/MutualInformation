Mutual Information
==================
This provides Mutual Information (MI) functions in Python. Currently, this provides the MI between tensors as described by [Kraskov et.al.](https://arxiv.org/pdf/cond-mat/0305641.pdf)


Mutual_information.py
---------------------

It takes in two numpy array files as command line inputs and outputs the mutual information in Mutual_information-k*.dat. The numpy array files should have same first index size, which is the data set count. Each of those data set can have any dimensional array (tensor).

Usage: 

```
./Mutual_information.py [-k<k value>] <numpy array file 1> <numpy array file 2> [<index for .dat file>]
```

Options:

    -k<k value>
        Number of nearest neighbours to account for, default=5
        
    <index for .dat file>
        number or index for easy plotting against Mutual information from .dat file


