import json
import time
import subprocess
import pandas as pd
import os
import sys
import random
from scipy.special import binom
from itertools import product
from functools import reduce

script_path = 'build_classifiers.py'

def get_real_time(script_path, config_path):
    start_time = time.time()
    subprocess.call(['python3', script_path, config_path], stdout=subprocess.DEVNULL)
    #subprocess.call(['python3', script_path, config_path])
    end_time = time.time()
    return end_time - start_time

def gen_config_with_n_k(n_k, origin_config_path, id_for_names = None):
    """Generate temporary config.json and n_k.csv with new n_k
    """
    if id_for_names == None:
        id_for_names = reduce(lambda x, y: str(x) + str(y), random.choices(range(10), k = 8)) # random id
    scout_n_k_name = 'n_k_{}.csv'.format(id_for_names)
    scout_config_name = 'config_{}.json'.format(id_for_names)
    
    config_dirname = os.path.dirname(origin_config_path)
    scout_n_k_path = os.path.join(config_dirname, scout_n_k_name)
    scout_config_path = os.path.join(config_dirname, scout_config_name)
    
    n_k = pd.DataFrame(n_k)
    n_k.to_csv(scout_n_k_path, index=False)
    
    config_path = open(origin_config_path, "r")
    config = json.load(config_path)
    config["n_k_path"] = scout_n_k_name
    with open(scout_config_path, 'w') as f:
        json.dump(config, f)
    
    return (scout_config_path, scout_n_k_path)

def lr_limits(P):
    """Find pair (z, z+1) for linear predicate P(x) such that
        P(x) = false for x <= z
        P(x) = true for x >= z+1
    """
    z = 1
    while not P(z):
        z += 1
    if z == 1:
        print("Little max working time", file=sys.stderr)
        sys.exit(1)
    return (z-1, z)

def gen_n_k(script_path, config_path,
            t, strategy, basis,
            f_time = lambda n, k: k * binom(n, k),
            n_k_scout = {"n": [7], "k": [5]}, c = None,
            output = sys.stdout):
    """Generate n_k grid by strategy from basis with working time t of script
    
    Parametors
    ----------
    script_path : string
        Path to script
    config_path : string
        Path to configuration file for script
    t : int or string
        Мax working time of script in seconds (TODO in format '%d%h%m%s')
    f_time : function int x int -> int
        f_time(n, k) -- asymptotic working time on n, k
    c : int
        Multiplier for f_time
    n_k_scout: {"n": [n_1, ..., n_m], "k": [k_1, ..., k_m]}
        Little grid for calculating multiplier c
    strategy : string
    basis : list
        Variants of strategy
            'k_to_n'
                Generate n_1, ..., n_p for basis = [k_1, ..., k_p]
            'n,>k'
                Generate (n, k), (n, k+1), ... (n, k+p) for basis = [n, k]
    """
    if c == None:
        tmp_config_path, tmp_n_k_path = gen_config_with_n_k(n_k_scout, config_path)
        real_time = get_real_time(script_path, tmp_config_path)
        os.remove(tmp_config_path)
        os.remove(tmp_n_k_path)
        eval_time = sum(map(lambda n, k: f_time(n, k), n_k_scout["n"], n_k_scout["k"]))
        c = real_time / eval_time
    print('Multiplier for asymptotic time function: {}'.format(c))
    
    if strategy == 'k_to_n':
        # Find [n_1, ..., n_p] with max closer time working to t
        t_0 = t / len(basis)
        n_vars = []
        for k in basis:
            n_l, n_r = lr_limits(lambda n: c * f_time(n, k) > t_0)
            n_vars.append((n_l, n_r))
        #n_vars = map(lambda k: lr_limits(lambda n: c * f_time(n, k) > t_0), basis) #[(n_l, n_r), ..., ]
        n_vars = list(product(*n_vars)) #[[n_1, ..., n_p], ...]
        nk_vars = list(map(lambda lt: list(zip(lt, basis)), n_vars)) #[[(n_1, k_1), ..., (n_p, k_p)], [(n'_1, k_1), ..., (n'_p, k_p)], ...]
        w_time = lambda gr: sum(list(map(lambda pr: c * f_time(pr[0], pr[1]), gr)))
        nk_vars = list(filter(lambda it: w_time(it) <= t, nk_vars))
        
        nk = max(nk_vars, key = w_time)
        n = list(map(lambda x: x[0], nk))
        k = basis
    elif strategy == 'n,>k':
        pass #TODO
    
    n_k = {"n": n, "k": k}
    #gen_config_with_n_k(n_k, config_path, id_for_names = 123)
    n_k_df = pd.DataFrame(n_k)
    n_k_df.to_csv(output, index=False)
    
    return n_k

if __name__ == "__main__":
    
    config_path = sys.argv[1]
    max_time = int(sys.argv[2])
    basis = list(map(int, sys.argv[3:]))
    
    #config_path = './examples/example1/config.json'
    #t = 60
    #basis = [5, 6]
    
    n_k = gen_n_k(script_path, config_path,
            t=max_time, strategy='k_to_n', basis=basis,
            #output = 'new_n_k.csv',
            #c = 0.8263466755549113
            )
