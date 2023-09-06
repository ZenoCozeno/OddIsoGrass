import numpy as np
import sage.libs.lrcalc.lrcalc as lrcalc

def formatting(v, k):
    array_v = np.array(v)
    padded = np.pad(array_v, (0, k-len(array_v)), mode='constant', constant_values=0)
    if not(np.array_equal(padded, np.sort(padded)[::-1])):
        raise Exception("Sorry, not correct")
    return padded



def LRfactors(U_alpha, U_beta, k):
    U_alpha = np.array(U_alpha)
    U_beta = np.array(U_beta)
    U_alpha_pos = U_alpha - U_alpha[-1] 
    U_beta_pos = U_beta - U_beta[-1]
    pos_factors_w_multip = lrcalc.mult(U_alpha_pos,U_beta_pos,k)
    factors_w_multip = {}
    for i in pos_factors_w_multip:
        factors_w_multip.update({tuple(formatting(i,k) + U_alpha[-1] + U_beta[-1]): pos_factors_w_multip[i]})
    return factors_w_multip