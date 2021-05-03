"""
Commonly used utils
"""

import numpy as np


def get_datasets(ann, datasets=None):
    """List of non-validation datasets

    Parameters
    ----------
    ann : pandas.DataFrame
        DataFrame with annotation of samples. This argument is
        actually not used by the function.
    datasets : array-like
        List of dataset identifiers which should be used to calculate
        test statistic. By default (None), union of all non-validation
        datasets will be used.

    Returns
    -------
    list
        List of non-validation datasets
    """
    if not datasets:
        datasets = np.unique(ann.loc[ann['Dataset type'] != 'Validation', 'Dataset'])

    return datasets
