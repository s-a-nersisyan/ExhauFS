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

from .regression.feature_selectors import *
from .classification.feature_selectors import *
from .wrappers import feature_selector_wrapper


@feature_selector_wrapper()
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


@feature_selector_wrapper()
def entity(df, ann, n):
    """Select first n features from a given DataFrame
    Returns
    -------
    list
        List of first n columns from a given DataFrame.
    """

    return list(df.columns)[:n]


@feature_selector_wrapper()
def median(df, ann, n):
    """Select n features with the highest median value
    Returns
    -------
    list
        List of n columns with highest median value.
    """
    return list(df.median().sort_values().index)[:n]
