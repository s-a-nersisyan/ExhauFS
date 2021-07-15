from scipy.stats import f_oneway

from src.core.utils import get_datasets


def f_test(df, ann, path_to_file, sep=None):
    """Pre-select features without difference between types of dataset

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
        List of features without difference between types of dataset intersected with a list of
        features from a given DataFrame.
    """
    datasets = get_datasets(ann)

    samples = [df.loc[ann['Dataset'] == dataset] for dataset in datasets]
    statistics, pvalues = f_oneway(*samples, axis=0)

    return df.columns[pvalues > 0.05].to_list()
