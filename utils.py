import sys
import os
import json

import pandas as pd

from core import classification
from core.regression import regression, regressors
from core import feature_pre_selectors
from core import feature_selectors
from core import preprocessors
from core.classification import classifiers
from core import accuracy_scores
from datetime import datetime


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
    df = pd.read_csv(os.path.join(config_dirname, config["data_path"]).replace("\\","/"), index_col=0)
    ann = pd.read_csv(os.path.join(config_dirname, config["annotation_path"]).replace("\\","/"), index_col=0)
    if load_n_k:
        n_k = pd.read_csv(os.path.join(config_dirname, config["n_k_path"]).replace("\\","/"))
    else:
        n_k = pd.DataFrame()
    output_dir = os.path.join(config_dirname, config["output_dir"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config["output_dir"] = output_dir
    
    # Ensure paths in config are relative to config directory
    if "path_to_file" in config.get("feature_pre_selector_kwargs", {}):
        correct_path = os.path.join(config_dirname, config["feature_pre_selector_kwargs"]["path_to_file"]).replace("\\","/")
        config["feature_pre_selector_kwargs"]["path_to_file"] = correct_path
    
    if "path_to_file" in config.get("feature_selector_kwargs", {}):
        correct_path = os.path.join(config_dirname, config["feature_selector_kwargs"]["path_to_file"]).replace("\\","/")
        config["feature_selector_kwargs"]["path_to_file"] = correct_path

    return config, df, ann, n_k


def initialize_classification_model(config, df, ann, n_k):
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
    classification.ExhaustiveClassification
        Initialized classification model.
    """

    # output directory
    output_dir = config["output_dir"] + "_" + \
                 str(datetime.now()).replace(" ", ".").replace(":", ".").replace("-", ".")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return classification.ExhaustiveClassification(
        df, ann, n_k, output_dir,
        getattr(feature_pre_selectors, config.get("feature_pre_selector") or "", None),
        config.get("feature_pre_selector_kwargs", {}),
        getattr(feature_selectors, config.get("feature_selector") or "", None),
        config.get("feature_selector_kwargs", {}),
        getattr(preprocessors, config["preprocessor"] or "", None),
        config["preprocessor_kwargs"],
        getattr(classifiers, config["classifier"]), 
        config["classifier_kwargs"],
        config["classifier_CV_ranges"], config["classifier_CV_folds"],
        config.get("limit_feature_subsets", False),
        config.get("n_feature_subsets", 0),
        config.get("shuffle_feature_subsets", False),
        {s: getattr(accuracy_scores, s) for s in config["scoring_functions"]},
        config["main_scoring_function"], config.get("main_scoring_threshold", 0.5),
        n_processes=config.get("n_processes", 1),
        random_state=config["random_state"],
        verbose=config.get("verbose", True)
    ), output_dir


def initialize_regression_model(config, df, ann, n_k):
    """Run the pipeline for regressor construction
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
    regression.ExhaustiveRegression
        Initialized regression model.
    """
    return regression.ExhaustiveRegression(
        df, ann,
        n_k=n_k,
        output_dir=config["output_dir"],
        feature_pre_selector=getattr(feature_pre_selectors, config.get("feature_pre_selector") or "", None),
        feature_pre_selector_kwargs=config.get("feature_pre_selector_kwargs", {}),
        feature_selector=getattr(feature_selectors, config.get("feature_selector") or "", None),
        feature_selector_kwargs=config.get("feature_selector_kwargs", {}),
        preprocessor=getattr(preprocessors, config["preprocessor"] or "", None),
        preprocessor_kwargs=config["preprocessor_kwargs"],
        model=getattr(regressors, config["regressor"]),
        model_kwargs=config["regressor_kwargs"],
        model_cv_ranges=config["regressor_CV_ranges"], model_cv_folds=config["regressor_CV_folds"],
        limit_feature_subsets=config.get("limit_feature_subsets", False),
        n_feature_subsets=config.get("n_feature_subsets", 0),
        shuffle_feature_subsets=config.get("shuffle_feature_subsets", False),
        scoring_functions={s: getattr(accuracy_scores, s) for s in config["scoring_functions"]},
        main_scoring_function=config["main_scoring_function"],
        main_scoring_threshold=config.get("main_scoring_threshold", 0.5),
        n_processes=config.get("n_processes", 1),
        random_state=config["random_state"],
        verbose=config.get("verbose", True)
    )
