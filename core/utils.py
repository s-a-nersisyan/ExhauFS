"""
Commonly used utils
"""

import inspect
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


def check_if_func_accepts_arg(func, arg):
    for param in inspect.signature(func).parameters:
        if param == arg:
            return True

    return False


def seconds_to_hours(seconds):
    return seconds / 3600


def float_to_latex(f):
    float_str = '{0:.2e}'.format(f)
    base, exponent = float_str.split('e')

    if 2 >= int(exponent) >= -2:
        return '{0:.3g}'.format(f)
    return r'{0} \times 10^{{{1}}}'.format(base, int(exponent))
