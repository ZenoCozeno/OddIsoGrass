from collections import defaultdict
import copy

def convert(weights):
    shrunk_weights = defaultdict(set)
    for i in weights:
        shrunk_weights[tuple([i[0]-i[1], i[2]-i[1]])].add(i[1])
    return shrunk_weights

def staircase(weight, k, n):
    a = weight[0]
    b = -weight[1]
    c =  2*n - k - a - b
    stair_cpx = defaultdict(set)
    for i in range(0, a+1):
        stair_cpx[tuple([a-i,-b])].add(0)
    for i in range(0, b):
        stair_cpx[tuple([i,i+1-b])].add(-i-1)
    for i in range(0, c+1):
        stair_cpx[tuple([b,-i])].add(-b-1)
    return stair_cpx

def evolvable(generated, cpx, w):
    contained = []
    for twist in range(w):
        admissible_twists = {}
        for element in cpx.keys():
            admissible_twists.update({element: set([j+twist for j in cpx[element]])})
        contained_twists = [admissible_twists[weight].issubset(generated[weight]) for weight in admissible_twists.keys()]
        if all(contained_twists):
            contained.append(twist)
    return contained

def apply_staircase(T,k,n):
    applied_staircase = {}
    w = 2*n+2-k
    D = copy.deepcopy(T)
    for t in T.keys():
        cpx = staircase(t,k,n)
        admissible_twists = evolvable(D, cpx, w)
        applied_staircase.update({t:admissible_twists})
        for twist in admissible_twists:
            for j in range(w+t[1]):
                D[tuple([j,t[1]])] = D[tuple([j,t[1]])].union(admissible_twists)
            for j in range(w+t[1]):
                D[tuple([-t[1],-j])] = D[tuple([-t[1],-j])].union(set([g-1+t[1] for g in admissible_twists]))
    for a in applied_staircase.keys():
        if len(applied_staircase[a])!= 0:
            print("Stair ",a," twists ", applied_staircase[a])
    for d in D.keys():
        if D[d] != T[d]:
            print(f"Staircase added at {d} the following twists {D[d] - T[d]}")
    return D