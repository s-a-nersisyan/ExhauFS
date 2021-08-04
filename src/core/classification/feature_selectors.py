import numpy as np
from scipy.stats import \
    spearmanr, \
    ttest_ind, \
    f_oneway

from src.core.wrappers import feature_selector_wrapper


@feature_selector_wrapper()
def f_test(df, ann, n):
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

    Returns
    -------
    list
        List of n features associated with the lowest p-values.
    """
    X = df.to_numpy()
    y = ann["Class"].to_numpy()

    statistics, pvalues = f_oneway(
        *[X[y == class_ind] for class_ind in np.unique(y)],
        axis=0,
    )
    features = df.columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][:n]


@feature_selector_wrapper()
def t_test(df, ann, n):
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
    Returns
    -------
    list
        List of n features associated with the lowest p-values.
    """
    X = df.to_numpy()
    y = ann["Class"].to_numpy()

    statistics, pvalues = ttest_ind(X[y == 0], X[y == 1], axis=0)
    features = df.columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][:n]


@feature_selector_wrapper()
def spearman_correlation(df, ann, n):
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
    Returns
    -------
    list
        List of n features associated with the highest absolute
        values of Spearman correlation.
    """
    X = df.to_numpy()
    y = ann["Class"].to_numpy()

    pvalues = [spearmanr(X[:, j], y).pvalue for j in range(X.shape[1])]
    features = df.columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][:n]
