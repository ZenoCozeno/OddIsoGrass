from utilities import LRfactors, formatting
from vanishing_odd import vanishingOddGrass
from collections import defaultdict

import numpy as np
import math


class complex:
    def __init__(self, entries):
        self.entries = entries 

    def display(self):
        for i in self.entries:
            self.entries[i].display()
            print(f"[{i}]", end= " ")
            print(" ")
    
    def shift(self, index):
        self.entries = {pos + index: entry for pos, entry in self.entries.items()}
    
    def stupid_truncation(self, rel_cutting_point):
        m = max(self.entries.keys())
        trunc_dx = complex({pos: entry for pos, entry in self.entries.items() if pos > m-rel_cutting_point})
        trunc_sx = complex({pos: entry for pos, entry in self.entries.items() if pos <= m-rel_cutting_point})
        return truncated_complex(trunc_dx, trunc_sx)
    
    def tensor(self, second_complex):
        partial_sum = defaultdict(complex_entry)
        for i in self.entries.keys():
            for j in second_complex.entries.keys():
                ij_summand = self.entries[i].tensor(second_complex.entries[j])
                partial_sum[i+j] = partial_sum[i+j].sum(ij_summand)
        tensored = complex(partial_sum)
        for i in tensored.entries.keys():
            tensored.entries[i].normal_form()
        return tensored
    
    def concentration(self):
        nonzero = [i for i in self.entries if len(self.entries[i].entry)!=0]
        return len(nonzero)
    
    def normal_form(self):
        avoidable=[]
        for i in self.entries:
            self.entries[i].normal_form()
            if all([j.mult == 0 for j in self.entries[i].entry]):
                avoidable.append(i)
        for p in avoidable:
            del self.entries[p]
                
    
    def higher_cohom_vanishing(self,k,n):
        nonvanish_cohoms = {i: self.entries[i].higher_cohom_vanishing(k,n) for i in self.entries}
        nonvanish_cohoms.normal_form()
        return complex(nonvanish_cohoms)
    
    def dual(self):
        dual_cpx={}
        for i in self.entries:
            dual_cpx.update({ -i : self.entries[i].dual()})
        return complex(dual_cpx)


class complex_summand:
    def __init__(self, weight, multiplicity):
        self.weight = tuple(weight)
        self.mult = multiplicity
    
    def display(self):
        print(f"{self.mult} * U^{self.weight}", end=" ")     

    def tensor(self, second_summand):
        product_summands_with_multip = LRfactors(self.weight, second_summand.weight, len(self.weight))
        flattened_product = []
        for summand in product_summands_with_multip:
            term = complex_summand(summand, self.mult*second_summand.mult * product_summands_with_multip[summand])
            flattened_product.append(term)
        total_entry = complex_entry(flattened_product)
        return total_entry
    
    def higher_cohom_vanishing(self,k,n):
        return vanishingOddGrass(self.weight,k,n)[0]
    
    def dual(self):
        opp = -np.array(self.weight)[::-1]
        dual_summand = complex_summand(opp, self.mult)
        return dual_summand
    
class complex_entry:
    def __init__(self, summands=[]):
        self.entry = summands
    
    def normal_form(self):
        weight_multip = defaultdict(int)
        for summand in self.entry:
            weight_multip[summand.weight] += summand.mult
        formatted_entries=[]
        for weight in weight_multip:
            if  weight_multip[weight] != 0:
                formatted_entries.append(complex_summand(weight, weight_multip[weight]))
        self.entry = formatted_entries

    def sum(self, second_term):
        total = complex_entry(self.entry + second_term.entry) 
        total.normal_form()
        return total

    def display(self):
        for i in self.entry:
            if i.mult > 0:
                    print(" + ", end=" "),
            i.display()

    def tensor(self, second_term):
        partial_sum = complex_entry()
        for i in self.entry:
            for j in second_term.entry:
                partial_sum = partial_sum.sum(i.tensor(j))
        return partial_sum
    
    def higher_cohom_vanishing(self,k,n):
        nonvanish = complex_entry()
        for summand in self.entry: 
            if not (summand.higher_cohom_vanishing(k,n)):
                nonvanish = nonvanish.sum(complex_entry([summand]))
        return nonvanish
    
    def dual(self):
        dual_entry = complex_entry()
        for i in self.entry:
            dual_entry = dual_entry.sum(complex_entry([i.dual()]))
        return dual_entry

    
class truncated_complex:
    def __init__(self, right_resol, left_resol):
        self.right = right_resol
        self.left = left_resol

    def partial_tensors(self, second_trunc_cpx,k,n):
        mixed_prods = [] 
        mixed_prods.append(self.left.tensor(second_trunc_cpx.left))
        mixed_prods.append(self.left.tensor(second_trunc_cpx.right))
        mixed_prods.append(self.right.tensor(second_trunc_cpx.left))
        mixed_prods.append(self.right.tensor(second_trunc_cpx.right))
        mixed_cohom = [i.higher_cohom_vanishing(k,n) for i in mixed_prods]
        min_cohom = np.argmin([i.concentration() for i in mixed_cohom])
        mixed_cohom[min_cohom].normal_form()
        return mixed_cohom[min_cohom]
        



def staircase(weight, k, n):
    weight =formatting(weight, k)
    stairs = {}
    for i in range(n+2-k):
        if weight[0] - i >=  weight[1]:
            stairs.update({n+1-k-i: complex_entry([complex_summand([weight[0] - i, weight[1], weight[2]], math.comb(n,i))])})
        elif weight[0] - i >=  weight[2]:
            stairs.update({n+1-k-i: complex_entry([complex_summand([weight[1] - 1, weight[0]-i, weight[2]], math.comb(n,i+1))])})
        else:
            stairs.update({n+1-k-i: complex_entry([complex_summand([weight[1] - 1, weight[2]-1, weight[0]-i], math.comb(n, i+2))])})
    final = complex(stairs)
    return final
