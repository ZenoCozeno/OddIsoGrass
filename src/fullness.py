from collections import defaultdict
import copy

# TODO: refactor code and implement staircase with the function,
#  avoid storing twists as dictionary and just have the bundles to generate as a list

"""
This section implements the method described in sec. 1.5 in the dissertation. 
"""

def convert(weights):
    """
        convert list of weights of length 3 in format U^{w1,w2,w3} to U^{w1-w2, 0, w3-w2}(w2). 
        We store list of bundles as a dictionary {(a,b): twists in the list}.

        Args:
            weights: a list of weights of len 3 in format U^{w1,w2,w3}

        Returns:
            shrunk_weights:  a defaultdict {(a,b): set(twists)}, such that U^{a,0,b}(twists) \in weights.
    """
    shrunk_weights = defaultdict(set)
    for i in weights:
        shrunk_weights[tuple([i[0]-i[1], i[2]-i[1]])].add(i[1])
    return shrunk_weights

def fullness_test(basis, k, n, max_iter = 20):
    """
        iterate through staircase rule and wedge product rule to see if it is possible to generate from a list of bundles (basis) 
        the set T in max iterations (def 20).
        Every iteration consists of: 
        - application of all possible staircase complex rules;
        - application of all possible symplectic relations rule;
        
        For n= 4, 5, max_iterations 10 is enough.

        Args:
            basis: a list of weights of len 3 in format U^{w1,w2,w3} to test
            k, n: fix IGr(k,2n+1), relevant for wedge power and staircase

        Returns:
            boolean: can I generate T from basis in iterations<max_iter
    """
    #Fano index
    w = 2*n + 1 - k
    generated = convert(basis)
    final_configuration = {} 
    #initialize final configuraion T= U^{i,0,-j} for i+j\leq w
    for i in range(w+1):
        for j in range(w+1-i):
            final_configuration.update({tuple([i,-j]):set(range(w+1))})
    for i in range(max_iter):
        if generated == final_configuration:
            return True
        print("Iteration", i)
        generated = apply_staircase(generated,k,n)
        generated = apply_wedge(generated,k,n)
        i+=1
    return False

def staircase(weight, k, n):
    """
        construction of truncated staircase complex for k=3 avoiding the last term

        Args:
            weight: a weight  in format U^{a,0,-b} to compute the staircase of
            k, n: fix IGr(k,2n+1), relevant for wedge power and staircase

        Returns:
            stairc_cpx: a dictionary of bundles and twists in the staircase complex.
    """
    # TODO: merge with complex.py staircase function
    a = weight[0]
    b = -weight[1]
    c =  2*n - k - a - b
    stair_cpx = defaultdict(set)
    for i in range(0, a+1):
        stair_cpx[tuple([a-i,-b])].add(0)
    for i in range(0, b):
        # use add and defaultdictionary because a term can appear twice with different twists
        stair_cpx[tuple([i,i+1-b])].add(-i-1)
    for i in range(0, c+1):
        stair_cpx[tuple([b,-i])].add(-b-1)
    return stair_cpx

def evolvable(generated, cpx, w):
    """
        determine which twists of a complex () are contained in a dictionary of bundles and twists 

        Args:
            generated: a defaultdict of bundles and twists 
            cpx: the complex we want to verify all terms belong to generate for some twists
            w: Fano index 2*n+1-k

        Returns:
            stairc_cpx: a dictionary of bundles and twists in the staircase complex.
    """
    contained = []
    for twist in range(w+1):
        admissible_twists = {}
        for element in cpx.keys():
            admissible_twists.update({element: set([j+twist for j in cpx[element]])})
        contained_twists = [admissible_twists[weight].issubset(generated[weight]) for weight in admissible_twists.keys()]
        if all(contained_twists):
            contained.append(twist)
    return contained

def apply_wedge(generated,k,n):
    """
    Apply symplectic relation rule. If a weight with a-b >= n+2-k and all other bundles with a'- b' <= a - b are contained in 
    generated for some twist, then we add it to U^{a,0,b} as well. 
    We print out the added terms

    Args:
        generated: defaultdict of weights and twists
        k,n: fix IGr(k,2n+1)

    Returns:
        added: dictionary of weights and twists after application of symplectic rule
        + print out the added terms
    """
    added = copy.deepcopy(generated)
    w = 2*n+1-k
    # iterate through the columns where we can apply symplectic relations
    for i in range(n+2-k, w+1):
        # iterate through every bundle with a+b=i
        # TODO: improve the iteration to avoid nested for
        for t in generated.keys():
            if t[0]-t[1] == i:
                common_part = set(range(w+1))
                # iterate through vector bundles with smaller equal a+b to determine the common twists that they all have
                for s in generated.keys():
                    if ((s[0]-s[1] <= i) and (s!=t)):
                        common_part = common_part.intersection(generated[s])
                # add the common twists
                added[t]=generated[t].union(common_part)
    # print out the added terms if any 
    for t in generated.keys():
        if added[t] != generated[t]:
            print(f"Using symplectic relations, we obtain for {t} the additional twists:", added[t] - generated[t])
    return added

def apply_staircase(generated,k,n):
    """
    TODO: to change w and understand
    Apply symplectic relation rule. If a weight with a-b >= n+2-k and all other bundles with a'- b' <= a - b are contained in 
    generated for some twist, then we add it to U^{a,0,b} as well. 
    We print out the added terms

    Args:
        generated: defaultdict of weights and twists
        k,n: fix IGr(k,2n+1)

    Returns:
        added: dictionary of weights and twists after application of symplectic rule
        + print out the added terms
    """
    applied_staircase = {}
    w = 2*n+2-k
    added = copy.deepcopy(generated)
    for t in generated.keys():
        before_staircase =  copy.deepcopy(added)
        cpx = staircase(t,k,n)
        admissible_twists = evolvable(added, cpx, w)
        for j in range(w+t[1]):
            added[tuple([j,t[1]])] = added[tuple([j,t[1]])].union(admissible_twists)
        for j in range(w+t[1]):
            added[tuple([-t[1],-j])] = added[tuple([-t[1],-j])].union(set([g-1+t[1] for g in admissible_twists if g-1+t[1]>=0]))
        if added != before_staircase:
            applied_staircase.update({t:admissible_twists})
    for a in applied_staircase.keys():
        if len(applied_staircase[a])!= 0:
            print("Stair ",a," twists ", applied_staircase[a])
    for d in added.keys():
        if added[d] != generated[d]:
            print(f"Staircase added at {d} the following twists {added[d] - generated[d]}")
    return added