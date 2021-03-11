# External imports
import sys
import os
import pandas as pd

# Internal imports
from utils import *

def main(config_path):
    
    config, df, ann, n_k = load_config_and_input_data(config_path, load_n_k = False)
    model = initialize_classification_model(config, df, ann, n_k)

    index = 0
    res = pd.DataFrame(columns=["n","k","Estimated time"])
    for n in range(1, model.max_n + 1):
        for k in range(1, n + 1):
            _, time = model.exhaustive_run_n_k(n, k, model.pre_selected_features)
            time = model.estimate_run_n_k_time(n, k, time)
            res.at[index, "n"] = n
            res.at[index, "k"] = k
            res.at[index, "Estimated time"] = time
            index += 1
            if time > model.max_estimated_time:
                break
    
    # Save results
    config_dirname = os.path.dirname(config_path)
    output_dir = os.path.join(config_dirname, config["output_dir"])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    res.to_csv("{}/estimated_times.csv".format(output_dir), index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])