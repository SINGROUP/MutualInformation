    I=pyMIestimator(X,Y,k)
    with open('Mutual_information-k'+str(k)+'.dat','a') as f:
        f.write('I('+arg[0]+';'+arg[1]+') = '+str(I)+(' '+arg[2] if len(arg)>2 else '')+'\n')