"""Functions for feature pre-selection

Every feature pre-selection function takes input DataFrame,
annotation DataFrame and some optional arguments, 
and returns a list of features which is
a subset of DataFrame column names.

Each feature pre-selection function have the following signature:
def feature_pre_selector(df, ann, **kwargs):
    # Code
    return list_of_features
"""


def from_file(df, ann, path_to_file, sep=None):
    """Pre-select features from a given file
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. This argument is
        actually not used by the function.
    path_to_file : str
        Path to a file which contains feature names in a first column.
        Path can be absolute or relative to the current working directory.
    sep : str
        Separator string using to identify first column in a given file.
        By default (None), any whitespace character will be used.

    Returns
    -------
    list
        List of features from a given file intersected with a list of
        features from a given DataFrame.
    """

    with open(path_to_file, "r") as f:
        features_from_file = [line.split(sep)[0] for line in f]

    return [feature for feature in features_from_file if feature in df.columns]
