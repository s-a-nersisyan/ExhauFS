import numpy as np
from scipy.stats import \
    spearmanr, \
    ttest_ind, \
    f_oneway

from src.core.utils import get_datasets


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
