import numpy as np
from collections import defaultdict, Counter
from typing import Tuple

from src.utilities import formatting, LRfactors

def vanishingEvenGrass(weight:np.array, k:int, n:int) -> bool:
    """
        compute cohomology of U^weight using the spectral sequence induced by the embedding in the even Grassmannian. 
        The entries in the resolution are given by wedge^i U^{0,\dots,-1}.
        Returns a dictionary of weight with relative position if the spectral sequence is not trivially zero. 

        Args:
            weight: a list of length k representing a weight
            k, n: fix IGr(k,2n+1)

        Returns:
            boolean: True if all entries are acyclic on the even Grassmannian
            nonvanish: a dictionary with shifts and nonvanishing entries
    """
    weight_padded = np.pad(weight, (0, n-len(weight)), mode='constant', constant_values=0)
    rho = np.array(range(n,0,-1))
    w = weight_padded + rho
    if 0 in set(w):
        return True 
    return not(len(set(np.abs(w)))==len(w))

def vanishingOddGrass(weight:np.array, k:int, n:int) -> Tuple[bool, dict]:
    """
        compute cohomology of U^weight using the spectral sequence induced by the embedding in the even Grassmannian. 
        The entries in the resolution are given by wedge^i U^{0,\dots,-1}.
        Returns a dictionary of weight with relative position if the spectral sequence is not trivially zero. 

        Args:
            weight: a list of length k representing a weight
            k, n: fix IGr(k,2n+1)

        Returns:
            boolean: True if all entries of the resolution are acyclic on the even Grassmannian
            nonvanish: a dictionary with shifts and nonvanishing entries
    """
    formatted_weight = formatting(weight, k)
    nonvanish = defaultdict(set)
    for i in range(k+1):
        wedge_i = [0]*(k-i) + [-1]*i
        prod_i = LRfactors(formatted_weight, wedge_i,k)
        for p_i in prod_i:
            if not(vanishingEvenGrass(p_i, k, n+1)):
                nonvanish[i].add(p_i)
    return len(nonvanish)==0, dict(nonvanish)
    
def extOddGrass(U_alpha: list, U_beta: list, k: int, n: int) -> Tuple[bool, dict]:
    """
        compute Ext(U_alpha, U_beta) using the vanishing on the odd Grassmannian. 
        The entries in the resolution are given by wedge^i U^{0,\dots,-1}.
        Returns a dictionary of non-acyclic weights and multiplicities. 

        Args:
            U_alpha, U_beta: as in Ext(U_alpha, U_beta)
            k, n: fix IGr(k,2n+1)
        Returns:
            boolean: True if all entries are acyclic 
            nonvanish: a Counter with nonvanishing entries and multiplicities. 
    """
    formatted_alpha = formatting(U_alpha, k)
    formatted_beta = formatting(U_beta, k)
    product = LRfactors(-formatted_alpha[::-1], formatted_beta, k)
    nonvanish = Counter({p: product[p] for p in product.keys() if not vanishingOddGrass(p, k, n)[0]})
    return len(nonvanish)==0, nonvanish

def Lefschetz_indep(U_alpha: list, U_beta: list, k: int, n: int) -> bool:
    """
        compute for which l we have Ext(U_alpha(l), U_beta) = 0 using the vanishing on the odd Grassmannian. 

        Args:
            U_alpha, U_beta: as in Ext(U_alpha, U_beta)
            k, n: fix IGr(k,2n+1)

        Returns:
            vanish_twists: the list of l=0,...,2n+1-k with Ext(U_alpha(l), U_beta) = 0
    """
    formatted_alpha = formatting(U_alpha, k)
    formatted_beta = formatting(U_beta, k)
    vanish_twists = [i for i in range(2*n+2-k) if extOddGrass(formatted_alpha+i, formatted_beta, k, n)[0]]
    return vanish_twists

def is_Lefschetz_excep(U_alpha: list, k: int, n: int) -> bool:
    """
        Does U_alpha satisfy: Ext(U_alpha(l), U_alpha) = 0 for l=1,...,2n+1-k or C if l=0?

        Args:
            U_alpha, k, n

        Return:
            a boolean that is true if the only non-acyclic term is U^[0,0,0] only if l = 0 with multiplicity 1.
    """
    formatted_alpha = formatting(U_alpha, k)
    if Lefschetz_indep(formatted_alpha, formatted_alpha, k, n) == [i for i in range(1, 2*n+2-k)]:
        non_vanishes = extOddGrass(formatted_alpha, formatted_alpha, k, n)[1]
        return non_vanishes == {tuple([0,0,0]): 1}
    else:
        return False
    
def is_Lefschetz_basis(except_sequence, k, n):
    """
        Is a sequence of weights a Lefschetz basis?

        Args:
            except_sequence: a list of weights presented as lists
            k, n

        Return:
            a boolean that is true if it is an exceptional basis, False 
            if either some Ext are nonvanishing or some weights are not exceptional.
    """
    for index in range(len(except_sequence)):
        if not is_Lefschetz_excep(except_sequence[index],k,n):
            print("not except:", except_sequence[index])
            return False
        for second_index in range(index+1, len(except_sequence)):
            if Lefschetz_indep(except_sequence[second_index], except_sequence[index], k, n) != list(range(2*n+2-k)):
                print("Problem with", except_sequence[second_index], except_sequence[index])
                return False
    return True

        


