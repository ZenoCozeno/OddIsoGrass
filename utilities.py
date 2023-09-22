import numpy as np
import sage.libs.lrcalc.lrcalc as lrcalc
from collections import Counter

def formatting(v: list, k: int) -> np.array:
    """
    from a list v, extend it with zeros until it has length k and check if it corresponds to a dominant weight, that is
    extends a list with zeros and raises an error if the list is not decreasing

    Args:
        v: vector to extend, any list-like data type
        k: extended length
    Return: 
        padded: a np.array of length k of decreasing entries
    """
    array_v = np.array(v, dtype=int)
    padded = np.pad(array_v, (0, k-len(array_v)), mode='constant', constant_values=0)
    if not(np.array_equal(padded, np.sort(padded)[::-1])):
        raise Exception("A weight should be decreasing")
    return padded

def LRfactors(U_alpha: list, U_beta: list, k:int) -> Counter:
    """
    computes the tensor product with multiplicities of two weights using LR-rule. 
    This is the key point that forces the use of Sage instead of Python.

    Args:
        U_alpha, U_beta: two weights
        k: as in IGr(k, 2n+1)
    Returns:
        factors_w_multip: a Counter object with entries weights and multiplicities coming from Littlewood-Richardson formula
    """
    U_alpha = formatting(U_alpha, k)
    U_beta = formatting(U_beta, k)
    #the function lrcalc works only on positive weights, hence we compute it on a twist
    U_alpha_pos = U_alpha - U_alpha[-1] 
    U_beta_pos = U_beta - U_beta[-1]
    pos_factors_w_multip = lrcalc.mult(U_alpha_pos,U_beta_pos,k)
    factors_w_multip = Counter({})
    #need to twist back again the entries and leave multiplicity unchanged
    for i in pos_factors_w_multip:
        factors_w_multip[tuple(formatting(i,k) + U_alpha[-1] + U_beta[-1])] += pos_factors_w_multip[i]
    return factors_w_multip

