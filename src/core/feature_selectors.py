"""Functions for feature selection

Every feature selection function takes input DataFrame, 
annotation DataFrame, number n and some optional arguments, 
and returns a list of n features which is
a subset of DataFrame column names.

Each feature selection function have the following signature:
def feature_selector(df, ann, n, **kwargs):
    # Code
    return n_element_list_of_features
"""

import numpy as np
from scipy.stats import \
    spearmanr, \
    ttest_ind, \
    f_oneway

from .utils import get_datasets

from .regression.feature_selectors import *

from pprint import pprint

def f_test(df, ann, n, datasets=None):
    """Select n features with the lowest p-values according to f-test

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. Three columns are mandatory:
        Class (binary labels), Dataset (dataset identifiers) and
        Dataset type (Training, Filtration, Validation).
    n : int
        Number of features to select.
    datasets : array-like
        List of dataset identifiers which should be used to calculate
        test statistic. By default (None), union of all non-validation
        datasets will be used.
    Returns
    -------
    list
        List of n features associated with the lowest p-values.
    """
    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann["Dataset"].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples]
    X = df_subset.to_numpy()
    y = ann_subset["Class"].to_numpy()

    statistics, pvalues = f_oneway(
        *[X[y == class_ind] for class_ind in np.unique(y)],
        axis=0,
    )
    features = df_subset.columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][:n]


def t_test(df, ann, n, datasets=None):
    """Select n features with the lowest p-values according to t-test
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. Three columns are mandatory:
        Class (binary labels), Dataset (dataset identifiers) and 
        Dataset type (Training, Filtration, Validation).
    n : int
        Number of features to select.
    datasets : array-like
        List of dataset identifiers which should be used to calculate
        test statistic. By default (None), union of all non-validation
        datasets will be used.
    feature : str
        Feature by witch to make hypothesis
    Returns
    -------
    list
        List of n features associated with the lowest p-values.
    """

    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann["Dataset"].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples]
    X = df_subset.to_numpy()
    y = ann_subset["Class"].to_numpy()

    statistics, pvalues = ttest_ind(X[y == 0], X[y == 1], axis=0)
    features = df_subset.columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][:n]


def spearman_correlation(df, ann, n, datasets=None):
    """Select n features with the highest correlation with target label
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. Three columns are mandatory:
        Class (binary labels), Dataset (dataset identifiers) and 
        Dataset type (Training, Filtration, Validation).
    n : int
        Number of features to select.
    feature : str
        Feature by witch to make hypothesis
    datasets : array-like
        List of dataset identifiers which should be used to calculate
        correlation. By default (None), union of all non-validation
        datasets will be used.

    Returns
    -------
    list
        List of n features associated with the highest absolute
        values of Spearman correlation.
    """

    datasets = get_datasets(ann, datasets)
    
    samples = ann.loc[ann["Dataset"].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples]
    X = df_subset.to_numpy()
    y = ann_subset["Class"].to_numpy()

    pvalues = [spearmanr(X[:, j], y).pvalue for j in range(X.shape[1])]
    features = df_subset.columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][:n]


def from_file(df, ann, n, path_to_file, sep=None):
    """Select first n features from a given file
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. This argument is
        actually not used by the function.
    n : int
        Number of features to select.
    path_to_file : str
        Path to a file which contains feature names in a first column.
        Path can be absolute or relative to the current working directory.
    sep : str
        Separator string using to identify first column in a given file.
        By default (None), any whitespace character will be used.

    Returns
    -------
    list
        List of first n entries in the intersection of features from a given file 
        with a list of features from a given DataFrame.
    """

    with open(path_to_file, "r") as f:
        features_from_file = [line.split(sep)[0] for line in f]
    
    return [feature for feature in features_from_file if feature in df.columns][:n]


def entity(df, ann, n, datasets=None):
    """Select first n features from a given DataFrame
    Returns
    -------
    list
        List of first n columns from a given DataFrame.
    """

    return list(df.columns)[:n]


def median(df, ann, n, datasets=None):
    """Select n features with the highest median value
    Returns
    -------
    list
        List of n columns with highest median value.
    """
    return list(df.median().sort_values().index)[:n]
