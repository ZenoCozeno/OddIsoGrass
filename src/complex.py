"""
This module contains the definitions of the following:

Objects:
    complex_entry: a Counter {weight:multiplicity}, where weight is  a tuple of decreasing integers
    complex: a dictionary {degree: complex_entry}
    truncated_complex: a pair of complexes that represent an object obtained performing a stupid truncation

Functions:
    staircase: the most relevant way to initialize a complex
"""

from src.utilities import LRfactors, formatting
from src.vanishing_odd import vanishingOddGrass
from collections import defaultdict, Counter

import numpy as np
import math

class complex:
    """
    This class represents a complex of the form ... -> \Sum mult * U^{weight} ->. 
    We ignore all the info regarding differentials of the complex, mostly being uniquely determined.
    Main way to initialize a complex object is through the function staircase. 

    Attributes: 
        degrees: a defaultdict object, with keys int representing the degrees and values complex_entry. 
                Default value the empty complex entry, corresponding to 0
    """
    def __init__(self, entries = {}):
        """
        Constructor for complex.

        Args: 
            entries: a dict represen with keys int representing the degrees and values complex_entry, 
                    representing the degrees and the complex entries.
        """
        self.degrees = defaultdict(complex_entry, entries) 

    def __str__(self):
        """
        str for complex class.

        Output:
            joined_cpx_list: a string with a representation of a complex. 
                            We avoid printing empty entries, except when the complex is trivial 
                            or the empty entry is between non-empty entries
        """
        nonzero = [deg for deg, entry in self.degrees.items() if len(entry.summands)!=0]
        #if the complex is all 0
        if len(nonzero)==0:
            return "0"
        cpx_str_list = []
        for i in range(min(nonzero), max(nonzero)+1):
            cpx_str_list.append(f"[{i}] " + str(self.degrees[i])) 
        joined_cpx_list = " ->\n".join(cpx_str_list)
        return "0 -> " + joined_cpx_list + " -> 0"
    
    def __add__(self, second_term):
        """
        sum for complex class. Sum complex entries of same degree
        """
        total = {}
        for i in set(self.degrees.keys()).union(set(second_term.degrees.keys())):
            #get instead of defaultdict to prevent creation of useless entries
            total.update({i: self.degrees.get(i, complex_entry()) + second_term.degrees.get(i, complex_entry())})
        return complex(total)
    
    def __mul__(self, second_complex):
        """
        tensor product for complex class. Tensor complex entries and place them in the appropriate degree (deg1+deg2).
        """
        partial_sum = complex()
        for i in self.degrees.keys():
            for j in second_complex.degrees.keys():
                ij_summand = self.degrees[i] * second_complex.degrees[j]
                partial_sum = partial_sum + complex({i+j: ij_summand})
        return partial_sum

    def shift(self, index):
        """
        shift of a complex, just change the grading
        """
        return complex({pos + index: entry for pos, entry in self.degrees.items()})

    def amplitude(self):
        """
        "naive amplitude" of a complex. 

        Return: 
            amp: max degree with nonzero entry - min degree with nonzero entry. If complex is zero, amp = 0.  
                If amp = 0, 1, it coincides with the true amplitude in cohomology.
        """
        nonzero = [i for i in self.degrees if len(self.degrees[i].summands)!=0]
        #if the complex is all 0
        if len(nonzero) == 0:
            amp = 0 
        else:
            amp = max(nonzero)+1 - min(nonzero) 
        return amp
    
    def non_vanish_terms(self,k,n):
        """
        Return:
            nonvanish_cohoms: subcomplex composed of non-acyclic terms in cohomology
        """
        nonvanish_cohoms = complex({deg: entry.non_vanish_terms(k,n) for deg, entry in self.degrees.items()})
        return nonvanish_cohoms
    
    def stupid_truncation(self, rel_cutting_point:int):
        """
        stupid truncation of a complex and its resolution.
        Args:
            self: an exact complex
            rel_cutting_point: an integer that says how many positions from the right we want to keep in the stupid truncation
        Return:
            a truncated complex: the right side has naive amplitude rel_cutting_point, 
                                the left side has naive amplitude self.amplitude() - rel_cutting_point, concentrated in degree 0
        """
        m = max(self.degrees.keys())
        trunc_dx = complex({pos - m + rel_cutting_point-1 : entry for pos, entry in self.degrees.items() if pos > m-rel_cutting_point})
        trunc_sx = complex({pos - m  + rel_cutting_point-1: entry for pos, entry in self.degrees.items() if pos <= m-rel_cutting_point})
        return truncated_complex(trunc_dx, trunc_sx)
    
    def dual(self):
        """
        dual complex. 

        Return: 
            dual_cpx: a complex with dual entries and opposite grading
        """
        dual_cpx={}
        for i in self.degrees:
            dual_cpx.update({ -i : self.degrees[i].dual()})
        return complex(dual_cpx)
    
    def cone(self, second_term):
        """
        cone of a morphism.
        """
        return self + second_term.shift(1)

class complex_entry:
    """
    This class represents a entry in a complex of the form \Sum mult * U^{weight} -  all in the same degree. 

    Attributes: 
        summands: a counter object of pairs {weight: multiplicity}, where weight is a tuple and multiplicity is an integer.
    """   
    def __init__(self, summands = {}) -> None:
        """
        Constructor for the complex_entry class.

        Args:
            a dictionary of {weight: multiplicity}, where weight is a tuple and multiplicity is an integer.
        """
        self.summands = Counter({k:c for k, c in summands.items() if c > 0} )
        
    def __str__(self) -> str:
        """
        str-method for the complex_entry class.

        Output:
            output: a formatted string to represent the entry of the complex. If the entry is trivial, the string equals "0"
        """
        if len(self.summands) >0 :
            str_list = [f"{self.summands[i]} * U^{i}" for i in self.summands]
            output = " + ".join(str_list)
        else:
            output = " 0 " 
        return output
    
    def __add__(self, second_term):
        """
        sum of entries. We sum multiplicites on common terms. If a term appears in only one complex_entry, 
        the multiplicity is unchanged
        """
        #+ is the + of Counter objects, which sums multiplicities and creates new entries
        total = self.summands + second_term.summands    
        return complex_entry(total)
        
    def __mul__(self, second_term):
        """
        tensor product of entries. We apply distributivity, while the multiplication of single weights is implemented in LRfactors
        """
        partial_sum = complex_entry({})
        for i in self.summands.keys():
            for j in second_term.summands.keys():
                product_summands_with_multip = LRfactors(i, j, len(i))
                #there is no multiplication by scalar for Counter objects, so we resort to this workaround
                for k in product_summands_with_multip.keys():
                    product_summands_with_multip[k] = product_summands_with_multip[k] * self.summands[i] * second_term.summands[j]
                partial_sum.summands = partial_sum.summands + product_summands_with_multip
        return partial_sum
    
    def dual(self):
        """
        dual entry. Every weight is replaced with the dual one, multiplicities unchanged.
        """
        dual_entry = {tuple(-np.array(i)[::-1]): self.summands[i] for i in self.summands}
        return complex_entry(dual_entry)
    
    def non_vanish_terms(self,k,n):
        """
        extract the terms of the sum which have nonzero cohomology iterating vanishingOddGrass
        """
        nonvanish = { weight: multip for weight, multip in self.summands.items() if (not vanishingOddGrass(weight, k,n)[0])}
        return complex_entry(nonvanish)
    
class truncated_complex:
    """
    truncated complex is made to store the result of a stupid_truncation, where we obtain a object with 2 
    derived-isomorphic resolutions. The most important methods used in the notebook are is_indep, shortest_Tor.

    Attributes:
        right: right resolution, stored in a complex object
        left: left resolution, stored in a complex object
    """
    def __init__(self, right_resol: complex, left_resol: complex):
        """
        Constructor for truncated_complex.

        Args: 
            right_resol: right resolution, stored in a complex object
            left_resol: left resolution, stored in a complex object
        """
        self.right = right_resol
        self.left = left_resol

    def __str__(self) -> str:
        """
        str for truncated_complex class.
        """
        return "Right: \n" + str(self.right) + "\nLeft: \n" + str(self.left) + "  end-cpx"
    
    def dual(self):
        """
        dual of the kernel of the right resolution. It is obtained dualizing the single resolutions and swapping them.
        """
        return truncated_complex(self.left.dual(), self.right.dual())

    def cone(self, second_trunc_cpx):
        """
        cone of the objects obtained as truncation of cones
        """
        return truncated_complex(self.right.cone(second_trunc_cpx.right), self.left.cone(second_trunc_cpx.left))
    
    def is_indep(self,   weights:list, k:int, n:int, verbose=False):
        """
        does the object T described by the truncated complex form an exact sequence with T, weights on IGr(k,2n+1)?

        Args:
            weights: a list of tuples
            k, n: data fixing the isotropic grassmannian IGr(k, 2n+1)
            verbose: if we want the list of weights with a nonzero Ext
        """
        problem_weights={}
        for i in weights:
            cpx = complex({0: complex_entry({tuple(i):1})})
            # this is a bit artificial
            trunc_cpx = truncated_complex(cpx, cpx)
            # if a complex .amplitude() is zero, then it must be zero (the opposite is not true and requires further study)
            if trunc_cpx.dual().shortest_Tor(self,k,n).amplitude() != 0:
                problem_weights.update({tuple(i):trunc_cpx.dual().shortest_Tor(self,k,n)})
        if verbose:
            if len(problem_weights) != 0:
                for weight, cohom in problem_weights.items():
                    print(f"{weight}:\n {str(cohom)}, \n")
        return len(problem_weights) == 0

    def shortest_Tor(self, second_trunc_cpx, k,n):
        """
        Given two truncated complexes, we determine which derived tensor product, 
        which can be computed in 2*2 ways with the adequate spectral sequence, 
        has the smallest amplitude of the non-acyclic part. 

        Args: 
            k, n: data fixing the isotropic grassmannian IGr(k, 2n+1)
        Return: 
            the non-acyclic part with smallest amplitude
        """
        mixed_prods = [] 
        mixed_prods.append(self.left * second_trunc_cpx.left)
        mixed_prods.append(self.left * second_trunc_cpx.right)
        mixed_prods.append(self.right * second_trunc_cpx.left)
        mixed_prods.append(self.right * second_trunc_cpx.right)
        mixed_cohom = [i.non_vanish_terms(k,n) for i in mixed_prods]
        min_cohom = np.argmin([i.amplitude() for i in mixed_cohom])
        return mixed_cohom[min_cohom]

def staircase(weight, k, m) -> complex:
    """
    Given a weight of length 3, satisfying weight[0]-weight[2] <= n-k, this returns the associate staircase complex 
    in the Grassmannian Gr(3, m).

    Args: 
        k, m: data fixing the grassmannian Gr(k, m)
    Return:
        the staicase complex as in Fonarev's work.
    """
    if k!= 3:
        raise Exception("k != 3 not implemented yet")
    # TODO: add k!=3
    weight =formatting(weight, k)
    if weight[0]-weight[-1] > m-k:
        raise Exception("staircase is defined only if weight[0]-weight[-1] is small enough")
    stairs = {}
    for i in range(m+2-k):
        if weight[0] - i >=  weight[1]:
            stairs.update({m+1-k-i: complex_entry({tuple([weight[0] - i, weight[1], weight[2]]): math.comb(m,i) })})
        elif weight[0] - i >=  weight[2]:
            stairs.update({m+1-k-i: complex_entry({tuple([weight[1] - 1, weight[0]-i, weight[2]]): math.comb(m,i+1) })})
        else:
            stairs.update({m+1-k-i: complex_entry({tuple([weight[1] - 1, weight[2]-1, weight[0]-i]): math.comb(m, i+2)})})
    final = complex(stairs)
    return final
