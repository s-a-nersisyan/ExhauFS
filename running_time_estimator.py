# External imports
import sys
import os
import pandas as pd

# Internal imports
from utils import *


def main(config_path, max_k, max_estimated_time, n_feature_subsets):
    config, df, ann, n_k = load_config_and_input_data(config_path, load_n_k=False)
    config["limit_feature_subsets"] = True
    config["shuffle_feature_subsets"] = True
    config["n_feature_subsets"] = n_feature_subsets

    model = initialize_classification_model(config, df, ann, n_k)

    res = pd.DataFrame(columns=["n", "k", "Estimated time"])
    for k in range(1, max_k + 1):
        for n in range(k, 3):#df.shape[1] + 1):
            _, time = model.exhaustive_run_n_k(n, k)
            time = model.estimate_run_n_k_time(n, k, time)
            if time > max_estimated_time:
                break

            res.loc[len(res)] = [n, k, time]

    res["n"] = res["n"].astype(int)
    res["k"] = res["k"].astype(int)

    # Save results
    output_dir = config["output_dir"]

    res.to_csv("{}/estimated_times.csv".format(output_dir), index=False)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    max_k = int(sys.argv[2])
    max_estimated_time = float(sys.argv[3])  # In hours
    n_feature_subsets = int(sys.argv[4])  # 100 is pretty good

    main(config_path, max_k, max_estimated_time, n_feature_subsets)
