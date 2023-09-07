import numpy as np
from collections import defaultdict

from utilities import formatting, LRfactors

def vanishingEvenGrass(weight, k, n):
    weight_padded = np.pad(weight, (0, n-len(weight)), mode='constant', constant_values=0)
    rho = np.array(range(n,0,-1))
    w = weight_padded + rho
    if 0 in set(w):
        return True 
    return not(len(set(np.abs(w)))==len(w))

def vanishingOddGrass(weight, k, n):
    formatted_weight = formatting(weight, k)
    nonvanish = defaultdict(set)
    for i in range(k+1):
        wedge_i = [0]*(k-i) + [-1]*i
        prod_i = LRfactors(formatted_weight, wedge_i,k)
        for p_i in prod_i:
            if not(vanishingEvenGrass(p_i, k, n+1)):
                nonvanish[i].add(p_i)
    return len(nonvanish)==0, dict(nonvanish)

    
def extOddGrass(U_alpha, U_beta, k, n):
    formatted_alpha = formatting(U_alpha, k)
    formatted_beta = formatting(U_beta, k)
    product = LRfactors(-formatted_alpha[::-1], formatted_beta, k)
    nonvanish = {p: product[p] for p in product.keys() if not vanishingOddGrass(p, k, n)[0]}
    return len(nonvanish)==0, nonvanish

def Lefschetz_indep(U_alpha, U_beta, k, n):
    formatted_alpha = formatting(U_alpha, k)
    formatted_beta = formatting(U_beta, k)
    vanish_twists = [i for i in range(2*n+2-k) if extOddGrass(formatted_alpha+i, formatted_beta, k, n)[0]]
    return vanish_twists

def is_Lefschetz_excep(U_alpha, k, n):
    formatted_alpha = formatting(U_alpha, k)
    if Lefschetz_indep(formatted_alpha, formatted_alpha, k, n) == [i for i in range(1, 2*n+2-k)]:
        non_vanishes = extOddGrass(formatted_alpha, formatted_alpha, k, n)[1]
        return non_vanishes == {tuple([0,0,0]): 1}
    else:
        return False
    
def is_Lefschetz_basis(except_sequence, k, n):
    for index in range(len(except_sequence)):
        if not is_Lefschetz_excep(except_sequence[index],k,n):
            print("not except")
            return False
        for second_index in range(index+1, len(except_sequence)):
            if Lefschetz_indep(except_sequence[second_index], except_sequence[index], k, n) != list(range(2*n+2-k)):
                print("Problem with", except_sequence[second_index], except_sequence[index])
                return False
    return True

        


