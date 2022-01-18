import numpy as np
from scipy.stats import \
    spearmanr, \
    ttest_ind, \
    f_oneway

from sklearn.linear_model import LogisticRegression

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


@feature_selector_wrapper()
def l1_logistic_regression(df, ann, n, C_low=0, C_high=1e+6, max_iter=1000):
    """Select n features with l1-penalized
    logistic regression model

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
    C_low: float
        Minimum inverse l1 penalizer value
    C_high: float
        Maximum inverse l1 penalizer value
    max_iter: int
        Maximum number of iterations before non-convergence error
    Returns
    -------
    list
        List of n features with non-zero coefficients
    """
    
    X = df.to_numpy()
    y = ann["Class"].to_numpy()
    
    if n > len(X):
        raise Exception("l1 logistic regression cannot select more features than a sample size")
    
    def select_features_from_model(model):
        non_zero_coef = np.abs(model.coef_[0]) >= 1e-5
        return df.columns.to_numpy()[non_zero_coef].tolist()
    
    model = LogisticRegression(
        penalty="l1", C=C_high,
        solver="liblinear", class_weight="balanced",
        warm_start=True, random_state=17
    )
    model.fit(X, y)

    C = None
    for i in range(max_iter):
        C_mid = (C_low + C_high) / 2
        model.set_params(C=C_mid)
        model.fit(X, y)
        n_mid = len(select_features_from_model(model))
        if n_mid == n:
            C = C_mid
            break
        elif n < n_mid:
            C_high = C_mid
        else:
            C_low = C_mid

    if C is None:
        raise Exception(f"Binary search failed to converge to n = {n} features")
    
    return select_features_from_model(model)
