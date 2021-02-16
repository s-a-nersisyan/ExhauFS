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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
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
    n_k = pd.read_csv(os.path.join(config_dirname, config["n_k_path"]))

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
    out = model.exhaustive_run()
    print(out)
    out.to_csv("test.csv")
