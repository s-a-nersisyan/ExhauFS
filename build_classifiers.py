import sys
import os

import pandas as pd

from utils import *

def main(config_path):
    
    # Load config and input data
    config, df, ann, n_k = load_config_and_input_data(config_path)
    
    # Build classifiers
    model = initialize_classification_model(config, df, ann, n_k)
    res = model.exhaustive_run()
    
    # Save raw results (classifiers and their quality scores)
    config_dirname = os.path.dirname(config_path)
    output_dir = os.path.join(config_dirname, config["output_dir"])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    res.to_csv("{}/classifiers.csv".format(output_dir))

    # Summary table #1: number of classifiers which passed
    # scoring threshold on training + filtration sets,
    # and training + filtration + validation sets
    summary_n_k = pd.DataFrame(columns=[
        "n", "k", "num_training_reliable", "num_validation_reliable", "percentage_reliable"
    ])
    for n, k in zip(n_k["n"], n_k["k"]):
        res_n_k = res.loc[(res["n"] == n) & (res["k"] == k)]
        # All classifiers already passed filtration on training and filtration datasets
        tf_num = len(res_n_k)

        # Now do filtration on training, filtration and 
        # validation datasets (i.e. all datasets)
        all_datasets = ann["Dataset"].unique()
        query_string = " & ".join([
            "(`{};{}` >= {})".format(
                ds, 
                config["main_scoring_function"], 
                config["main_scoring_threshold"]
            ) for ds in all_datasets
        ])
        all_num = len(res_n_k.query(query_string))

        summary_n_k = summary_n_k.append({
            "n": n, "k": k,
            "num_training_reliable": tf_num,
            "num_validation_reliable": all_num,
            "percentage_reliable": all_num / tf_num * 100 if tf_num != 0 else 0
        }, ignore_index=True)
    
    summary_n_k["n"] = summary_n_k["n"].astype(int)
    summary_n_k["k"] = summary_n_k["k"].astype(int)
    summary_n_k["num_training_reliable"] = summary_n_k["num_training_reliable"].astype(int)
    summary_n_k["num_validation_reliable"] = summary_n_k["num_validation_reliable"].astype(int)
    summary_n_k.to_csv("{}/summary_n_k.csv".format(output_dir), index=None)
    
    # Summary table #2: for each feature calculate
    # percentage of reliable classifiers which use it
    feature_counts = {}
    for features_subset in res.index:
        for feature in features_subset.split(";"):
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    summary_features = pd.DataFrame(
        {"percentage_classifiers": feature_counts.values()},
        index=feature_counts.keys()
    )
    summary_features["percentage_classifiers"] *= 100 / len(res) if len(res) else 1
    summary_features = summary_features.sort_values("percentage_classifiers", ascending=False)
    summary_features.index.name = "gene"

    summary_features.to_csv("{}/summary_features.csv".format(output_dir))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])