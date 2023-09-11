from utilities import LRfactors, formatting
from vanishing_odd import vanishingOddGrass
from collections import defaultdict, Counter, namedtuple

import numpy as np
import math


class complex:
    def __init__(self, entries = {}):
        self.degrees = defaultdict(complex_entry, entries) 

    def __str__(self):
        cpx_str_list = [f"[{i}] " + str(self.degrees[i]) for i in self.degrees] 
        joined_cpx_list = " <-\n".join(cpx_str_list)
        return "0 <- " + joined_cpx_list + " <- 0"
    
    def __add__(self, second_term):
        total = {}
        for i in set(self.degrees.keys()).union(set(second_term.degrees.keys())):
            total.update( {i: self.degrees[i] + second_term.degrees[i]})
        return complex(total)
    
    def __mul__(self, second_complex):
        partial_sum = complex()
        for i in self.degrees.keys():
            for j in second_complex.degrees.keys():
                ij_summand = self.degrees[i] * second_complex.degrees[j]
                partial_sum = partial_sum + complex({i+j: ij_summand})
        return partial_sum

    def shift(self, index):
        self.degrees = {pos + index: entry for pos, entry in self.degrees.items()}
    
    def amplitude(self):
        nonzero = [i for i in self.degrees if len(self.degrees[i].summands)!=0]
        if len(nonzero) == 0:
            amp = 0 
        else:
            amp = max(nonzero) - min(nonzero) 
        return amp
    
    def non_vanish_terms(self,k,n):
        nonvanish_cohoms = complex({deg: entry.non_vanish_terms(k,n) for deg, entry in self.degrees.items()})
        return nonvanish_cohoms
    
    def stupid_truncation(self, rel_cutting_point):
        m = max(self.degrees.keys())
        trunc_dx = complex({pos: entry for pos, entry in self.degrees.items() if pos > m-rel_cutting_point})
        trunc_sx = complex({pos: entry for pos, entry in self.degrees.items() if pos <= m-rel_cutting_point})
        return truncated_complex(trunc_dx, trunc_sx)
    
    def dual(self):
        dual_cpx={}
        for i in self.degrees:
            dual_cpx.update({ -i : self.degrees[i].dual()})
        return complex(dual_cpx)

class complex_entry:
    """
        complex_entry represents a entry in a complex of the form \Sum mult * U^{weight} -  all in the same degree,
        defaulted to 0. A entry is recorded as a counter object with entries (weight: multiplicity)
        Methods: 
            init: from a dictionary of couples (weight: multiplicity)
            add: sum of two bundles with multiplicities, returns a complex
            str: represents the bundle as printable
            mul: tensor product of two bundles
            dual: dualization of a bundle

    """   
    def __init__(self, summands = {}):
        self.summands = Counter({k:c for k, c in summands.items() if c > 0} )
        
    def __str__(self):
        if len(self.summands) >0 :
            str_list = [f"{self.summands[i]} * U^{i}" for i in self.summands]
            output = " + ".join(str_list)
        else:
            output = " 0 " 
        return output
    
    def __add__(self, second_term):
        total = self.summands + second_term.summands
        return complex_entry(total)
        
    def __mul__(self, second_term):
        partial_sum = complex_entry({})
        for i in self.summands.keys():
            for j in second_term.summands.keys():
                product_summands_with_multip = LRfactors(i, j, len(i))
                for k in product_summands_with_multip.keys():
                    product_summands_with_multip[k] = product_summands_with_multip[k] * self.summands[i] * second_term.summands[j]
                partial_sum.summands = partial_sum.summands + product_summands_with_multip
        return partial_sum
    
    def dual(self):
        dual_entry = {tuple(-np.array(i)[::-1]): self.summands[i] for i in self.summands}
        return complex_entry(dual_entry)
    
    def non_vanish_terms(self,k,n):
        nonvanish = { weight: multip for weight, multip in self.summands.items() if (not vanishingOddGrass(weight, k,n)[0])}
        return complex_entry(nonvanish)
    
class truncated_complex:
    def __init__(self, right_resol, left_resol):
        self.right = right_resol
        self.left = left_resol

    def partial_tensors(self, second_trunc_cpx,k,n):
        mixed_prods = [] 
        mixed_prods.append(self.left * second_trunc_cpx.left)
        mixed_prods.append(self.left * second_trunc_cpx.right)
        mixed_prods.append(self.right * second_trunc_cpx.left)
        mixed_prods.append(self.right * second_trunc_cpx.right)
        mixed_cohom = [i.non_vanish_terms(k,n) for i in mixed_prods]
        min_cohom = np.argmin([i.amplitude() for i in mixed_cohom])
        return mixed_cohom[min_cohom]
        



def staircase(weight, k, n):
    weight =formatting(weight, k)
    stairs = {}
    for i in range(n+2-k):
        if weight[0] - i >=  weight[1]:
            stairs.update({n+1-k-i: complex_entry({tuple([weight[0] - i, weight[1], weight[2]]): math.comb(n,i) })})
        elif weight[0] - i >=  weight[2]:
            stairs.update({n+1-k-i: complex_entry({tuple([weight[1] - 1, weight[0]-i, weight[2]]): math.comb(n,i+1) })})
        else:
            stairs.update({n+1-k-i: complex_entry({tuple([weight[1] - 1, weight[2]-1, weight[0]-i]): math.comb(n, i+2)})})
    final = complex(stairs)
    return final
