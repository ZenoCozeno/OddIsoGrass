import numpy as np

from utilities import formatting

        

def vanishingEven(u, k, n):
    revu = list(u) +([0] * (n+1 - len(u)))
    v = list(range(n+1,0,-1))
    w = sum(v, revu)
    if 0 in set(w):
        return True
    wneg = diff([0]*len(w),w) 
    vals = w + wneg
    return not(len(set(vals))==len(vals))

def spectrSequence(u:np.array, k,n):
    u=list(u)+[0]*(k-len(u))
    for i in range(k+1):
        wed_i = [0]*i + [-1]*(k-i)
        prod_i = goodLRcalc(u,wed_i,k)
        for p_i in prod_i:
            if not(vanishingEven(p_i, k, n)):
                return False
    return True

