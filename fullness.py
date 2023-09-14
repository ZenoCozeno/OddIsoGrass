from collections import defaultdict
from complex import staircase

def convert(weights):
    shrunk_weights = defaultdict()
    for i in weights:
        shrunk_weights[tuple(i[0]-i[1], i[2]-i[1])].add(i[1])
    return shrunk_weights

def staircase_rule(weight, k, n):
    cpx = staircase(weight, k, 2*n+1)
    M = min(cpx.items())
    del cpx[min]
    cpx_formatted = convert(cpx)
    return cpx_formatted