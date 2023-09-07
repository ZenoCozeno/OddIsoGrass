import numpy as np
from collections import defaultdict

from utilities import formatting, LRfactors

def vanishingEvenGrass(weight, k, n):
    weight_padded = np.pad(weight, (0, n-len(weight)), mode='constant', constant_values=0)
    rho = np.array(range(n,0,-1))
    w = weight_padded + rho
    if 0 in set(w):
        return True 
    return not(len(set(w))==len(np.abs(w)))

def vanishingOddGrass(weight, k, n):
    formatted_weight = formatting(weight, k)
    nonvanish = defaultdict(set)
    for i in range(k+1):
        wedge_i = [0]*(k-i) + [-1]*i
        prod_i = LRfactors(formatted_weight, wedge_i,k)
        for p_i in prod_i:
            if not(vanishingEvenGrass(p_i, k, n+1)):
                nonvanish[i].add(p_i)
    if len(nonvanish)==0:
        return True, dict(nonvanish)
    else:
        return False, dict(nonvanish)

