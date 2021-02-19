import sys
import os
import json

import pandas as pd

from core import classification  
from core import feature_pre_selectors
from core import feature_selectors
from core import preprocessors
from core import classifiers
from core import accuracy_scores


def load_config_and_input_data(config_path, load_n_k=True):
    """Load configuration file and input data
    
    Parameters
    ----------
    config_path : string
        Path to config file (json).
    load_n_k : bool
        Whether load n_k table or not.

    Returns
    -------
    dict, pd.DataFrame, pd.DataFrame, pd.DataFrame
        Dict with configuration, data (df), annotation (ann) and 
        table with n, k pairs (n_k).
    """

    try:
        config_file = open(config_path, "r")
    except:
        print("Cannot open configuration file", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.load(config_file)
    except:
        print("Please specify valid json configuration file", file=sys.stderr)
        sys.exit(1)

    # Paths are absolute or relative to config file
    # TODO: add some checks (files can be opened, indices are the same, columns names are here etc)
    config_dirname = os.path.dirname(config_path)
    df = pd.read_csv(os.path.join(config_dirname, config["data_path"]), index_col=0)
    ann = pd.read_csv(os.path.join(config_dirname, config["annotation_path"]), index_col=0)
    if load_n_k:
        n_k = pd.read_csv(os.path.join(config_dirname, config["n_k_path"]))
    else:
        n_k = pd.DataFrame()

    return config, df, ann, n_k


def run_exhaustive_classification(config, df, ann, n_k):
    """Run the pipeline for classifier construction
    using exhaustive feature selection.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. Three columns are mandatory:
        Class (binary labels), Dataset (dataset identifiers) and 
        Dataset type (Training, Filtration, Validation).
    n_k : pandas.DataFrame
        DataFrame with columns n and k defining a grid
        for exhaustive feature selection: n is a number
        of selected features, k is a length of each
        features subset.

    Returns
    -------
    pandas.DataFrame
        DataFrame with constructed classifiers and their
        quality scores.
    """
    model = classification.ExhaustiveClassification(
        df, ann, n_k,
        getattr(feature_pre_selectors, config["feature_pre_selector"] or "", None),
        config["feature_pre_selector_kwargs"],
        getattr(feature_selectors, config["feature_selector"]),
        config["feature_selector_kwargs"],
        getattr(preprocessors, config["preprocessor"] or "", None),
        config["preprocessor_kwargs"],
        getattr(classifiers, config["classifier"]), 
        config["classifier_kwargs"],
        config["classifier_CV_ranges"], config["classifier_CV_folds"],
        {s: getattr(accuracy_scores, s) for s in config["scoring_functions"]},
        config["main_scoring_function"], config["main_scoring_threshold"],
        n_processes=config["n_processes"], 
        random_state=config["random_state"],
        verbose=config["verbose"]
    )
    return model.exhaustive_run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)
    
    # Load config and input data
    config_path = sys.argv[1]
    config, df, ann, n_k = load_config_and_input_data(config_path)
    
    # Build classifiers
    res = run_exhaustive_classification(config, df, ann, n_k)
    
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
    summary_n_k["percentage_reliable"] = summary_n_k["percentage_reliable"].astype(int)
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
