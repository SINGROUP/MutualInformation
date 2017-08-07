Mutual Information between Natural Numbers and their squared values
===================================================================

In this test we'll use MutualInformation.py to find mutual information between natural numbers and their squared values.

```
./MutualInformation.py Natural_numbers-500.npy Natural_numbers_squared-500.npy 500\ natural\ numbers
./MutualInformation.py Natural_numbers-1000.npy Natural_numbers_squared-1000.npy 1000\ natural\ numbers
```

This gives the mutual information values in "Mutual_information-k5.dat":
    I(500) = 4.23199009666
    I(1000) = 4.92498752722
    
Using properties for mutual information:
----------------------------------------

Theoretical maximum:
```python
MI.digamma(500)-MI.digamma(6)
>>> 4.50749009666
MI.digamma(1000)-MI.digamma(6)
>>> 5.20113752722
```

Transferring mutual information for 500 instances to 1000 instances to compare:

'''python
I_500 + MI.digamma(1000) - MI.digamma(500)
>>> 4.9256375272198225
'''

This value is very close to the mutual information with 1000 instances. Implying transfering to same instances is important to compare and it works
