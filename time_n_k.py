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

class TimePredicter:
    """

    Parametors
    ----------
    script_path : string
        Path to script
    config_path : string
        Path to configuration file for script
    c : int (optional)
        Multiplier for time asymptotic function
    n_k_scout: {"n": [n_1, ..., n_m], "k": [k_1, ..., k_m]}
        Little grid for calculating multiplier c
    """
    
    def __init__(self, script_path, config_path, c = None, n_k_scout = {"n": [4], "k": [3]}):
        self.script_path = script_path
        self.config_path = config_path
        self.config = self.read_config()
        self.f_time = self.get_time_asymptotic_function()
        self.n_k_scout = n_k_scout 
        self.c = c if c is not None else self.get_time_modifier_by_scout()
        print('Multiplier for asymptotic time function: {}'.format(self.c))
        
    def read_config(self):
        config_file = open(self.config_path, "r")
        config = json.load(config_file)
        config_file.close()
        return config
    
    def get_time_asymptotic_function(self):
        return lambda n, k: k * binom(n, k)
    
    def get_time_modifier_by_scout(self):
        tmp_config_path, tmp_n_k_path = self.gen_modified_config(modified_n_k = self.n_k_scout)
        real_time = self.get_script_real_time(new_config_path = tmp_config_path)
        os.remove(tmp_config_path)
        os.remove(tmp_n_k_path)
        estimated_time = sum(map(self.f_time, self.n_k_scout["n"], self.n_k_scout["k"]))
        c = real_time / estimated_time
        return c
    
    def gen_modified_config(self, modified_n_k, config_filename = None, modified_n_k_filename = None):
        
        id_for_names = reduce(lambda x, y: str(x) + str(y), random.choices(range(10), k = 8)) # random id
        if modified_n_k_filename is None:
            modified_n_k_filename = 'n_k_{}.csv'.format(id_for_names)
        if config_filename is None:
            config_filename = 'config_{}.json'.format(id_for_names)
        
        config_dirname = os.path.dirname(self.config_path)
        modified_n_k_path = os.path.join(config_dirname, modified_n_k_filename)
        config_path = os.path.join(config_dirname, config_filename)
        
        n_k_df = pd.DataFrame(modified_n_k)
        n_k_df.to_csv(modified_n_k_path, index=False)
        
        config = self.config.copy()
        config["n_k_path"] = modified_n_k_filename
        with open(config_path, 'w') as f:
            json.dump(config, f)
        return (config_path, modified_n_k_path)
    
    def get_script_real_time(self, new_config_path = None, verbose = False):
        config_path = self.config_path if new_config_path is None else new_config_path
        start_time = time.time()
        if verbose:
            subprocess.call(['python3', self.script_path, config_path])
        else:
            subprocess.call(['python3', self.script_path, config_path], stdout=subprocess.DEVNULL)
        end_time = time.time()
        return end_time - start_time
    
    def gen_time_table(self, size = 10, output = sys.stdout):
        """
        Generate table with estimated working time (in seconds) of script for all n, k <= size
        
        n\k 0 1 2 ... 
        0   . . .
        1   . . . 
        2   . . .
        .   . .
        """
        
        def it(n, k): return 0 if k > n or k == 0 or n == 0 else int(self.c * self.f_time(n, k))
    
        tb_df = pd.DataFrame([[it(n, k) for k in range(size)] for n in range(size)])
        tb_df.to_csv(output)
        return tb_df
    
    def gen_n_k(self, t, strategy, basis, output_n_k = sys.stdout):
        """Generate n_k grid by strategy from basis with working time t of script
        
        Parametors
        ----------
        t : int or string
            Мax working time of script in seconds (TODO in format '%d%h%m%s')
        strategy : string
        basis : list
            Variants of strategy
                'k_to_n'
                    Generate n_1, ..., n_p for basis = [k_1, ..., k_p]
                'n,>k'
                    Generate (n, k), (n, k+1), ... (n, k+p) for basis = [n, k]
        """
        if strategy == 'k_to_n':
            # Find [n_1, ..., n_p] with max closer time working to t
            t_0 = t / len(basis)
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
            n_vars = []
            for k in basis:
                n_l, n_r = lr_limits(lambda n: self.c * self.f_time(n, k) > t_0)
                n_vars.append((n_l, n_r))
            #n_vars = map(lambda k: lr_limits(lambda n: c * f_time(n, k) > t_0), basis) #[(n_l, n_r), ..., ]
            n_vars = list(product(*n_vars)) #[[n_1, ..., n_p], ...]
            nk_vars = list(map(lambda lt: list(zip(lt, basis)), n_vars)) #[[(n_1, k_1), ..., (n_p, k_p)], [(n'_1, k_1), ..., (n'_p, k_p)], ...]
            w_time = lambda gr: sum(list(map(lambda pr: self.c * self.f_time(pr[0], pr[1]), gr)))
            nk_vars = list(filter(lambda it: w_time(it) <= t, nk_vars))
            
            nk = max(nk_vars, key = w_time)
            n_k = {"n": list(map(lambda x: x[0], nk)), "k": basis}
        elif strategy == 'n,>k':
            pass #TODO
        
        n_k_df = pd.DataFrame(n_k)
        n_k_df.to_csv(output, index=False)
        
        return n_k

if __name__ == "__main__":
    config_path = sys.argv[1]
    #config_path = './examples/example1/config.json'
    size = int(sys.argv[2])
    
    tm = TimePredicter(script_path, config_path)
    tm.gen_time_table(size=10)
