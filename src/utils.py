import sys
import os
import json
from functools import partial
from shutil import copyfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import binom, rankdata

from src.core.classification import classifiers, classification
from src.core.regression import regressors, regression
from src.core import feature_pre_selectors
from src.core import preprocessors
from src.core import accuracy_scores, feature_selectors
from datetime import datetime


def save_plt_fig(name, format):
    if format == 'tiff':
        kwargs = {'compression': 'tiff_lzw'} if format == 'tiff' else None
        plt.savefig(name, format=format, pil_kwargs=kwargs, dpi=350)
    else:
        plt.savefig(name, format=format, dpi=350)


def getattr_with_kwargs(module, method):
    if isinstance(method, dict):
        return partial(getattr(module, method['name']), **method.get('kwargs', {}))

    return getattr(module, method)


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

    print('Loading data...')
    try:
        config_file = open(config_path, 'r')
    except:
        print('Cannot open configuration file', file=sys.stderr)
        sys.exit(1)

    try:
        config = json.load(config_file)
    except:
        print('Please specify valid json configuration file', file=sys.stderr)
        sys.exit(1)

    # Paths are absolute or relative to config file
    config_dirname = os.path.dirname(config_path)
    df = pd.read_csv(os.path.join(config_dirname, config['data_path']).replace('\\','/'), index_col=0)
    ann = pd.read_csv(os.path.join(config_dirname, config['annotation_path']).replace('\\','/'), index_col=0)
    if load_n_k:
        n_k = pd.read_csv(os.path.join(config_dirname, config['n_k_path']).replace('\\','/'))
    else:
        n_k = pd.DataFrame()
    output_dir = os.path.join(config_dirname, config['output_dir'])
    # output directory
    output_dir = f"{output_dir}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    config['output_dir'] = output_dir
    # save config for further analysis
    copyfile(config_path, os.path.join(output_dir, 'config.json'))

    # Ensure paths in config are relative to config directory
    if 'path_to_file' in config.get('feature_pre_selector_kwargs', {}):
        correct_path = os.path.join(config_dirname, config['feature_pre_selector_kwargs']['path_to_file']).replace('\\','/')
        config['feature_pre_selector_kwargs']['path_to_file'] = correct_path

    if 'path_to_file' in config.get('feature_selector_kwargs', {}):
        correct_path = os.path.join(config_dirname, config['feature_selector_kwargs']['path_to_file']).replace('\\','/')
        config['feature_selector_kwargs']['path_to_file'] = correct_path
    print('Loaded data...')

    df = df[df.index.isin(ann.index)]
    ann = ann[ann.index.isin(df.index)]

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

    return classification.ExhaustiveClassification(
        df=df,
        ann=ann,
        n_k=n_k,
        output_dir=config['output_dir'],
        feature_pre_selector=getattr(feature_pre_selectors, config.get('feature_pre_selector') or '', None),
        feature_pre_selector_kwargs=config.get('feature_pre_selector_kwargs', {}),
        feature_selector=getattr(feature_selectors, config.get('feature_selector') or '', None),
        feature_selector_kwargs=config.get('feature_selector_kwargs', {}),
        preprocessor=getattr(preprocessors, config['preprocessor'] or '', None),
        preprocessor_kwargs=config['preprocessor_kwargs'],
        model=getattr(classifiers, config['model']),
        model_kwargs=config.get('model_kwargs', {}),
        model_cv_ranges=config.get('model_CV_ranges', []),
        model_cv_folds=config.get('model_CV_folds', 0),
        scoring_functions={s: getattr_with_kwargs(accuracy_scores, s) for s in config['scoring_functions']},
        main_scoring_function=config['main_scoring_function'],
        main_scoring_threshold=config.get('main_scoring_threshold', 0.5),
        limit_feature_subsets=config.get('limit_feature_subsets', False),
        n_feature_subsets=config.get('n_feature_subsets', 0),
        shuffle_feature_subsets=config.get('shuffle_feature_subsets', False),
        n_processes=config.get('n_processes', 1),
        random_state=config['random_state'],
        verbose=config.get('verbose', True),
    )


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
        df=df,
        ann=ann,
        n_k=n_k,
        output_dir=config['output_dir'],
        feature_pre_selector=getattr(feature_pre_selectors, config.get('feature_pre_selector') or '', None),
        feature_pre_selector_kwargs=config.get('feature_pre_selector_kwargs', {}),
        feature_selector=getattr(feature_selectors, config.get('feature_selector') or '', None),
        feature_selector_kwargs=config.get('feature_selector_kwargs', {}),
        preprocessor=getattr(preprocessors, config['preprocessor'] or '', None),
        preprocessor_kwargs=config['preprocessor_kwargs'],
        model=getattr(regressors, config['model']),
        model_kwargs=config.get('model_kwargs', {}),
        model_cv_ranges=config.get('model_CV_ranges', []),
        model_cv_folds=config.get('model_CV_folds', 0),
        scoring_functions={s: getattr_with_kwargs(accuracy_scores, s) for s in config['scoring_functions']},
        main_scoring_function=config['main_scoring_function'],
        main_scoring_threshold=config.get('main_scoring_threshold', 0.5),
        limit_feature_subsets=config.get('limit_feature_subsets', False),
        n_feature_subsets=config.get('n_feature_subsets', 0),
        shuffle_feature_subsets=config.get('shuffle_feature_subsets', False),
        n_processes=config.get('n_processes', 1),
        random_state=config['random_state'],
        verbose=config.get('verbose', True),
    )
