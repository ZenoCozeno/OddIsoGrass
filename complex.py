from utilities import LRfactors, formatting
from collections import defaultdict

import numpy as np
import math


class truncated_complex:
    def __init__(self, right_resol, left_resol):
        self.right = right_resol
        self.left = left_resol

class complex_summand:
    def __init__(self, weight, multiplicity):
        self.weight = weight
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

def staircase(weight, k, n):
    weight = formatting(weight, k)
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
