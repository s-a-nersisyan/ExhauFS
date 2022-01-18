import numpy as np

from src.core.regression.models import CoxRegression
from src.core.wrappers import feature_selector_wrapper


@feature_selector_wrapper()
def l1_cox(df, ann, n, p_low=0, p_high=1e+6, max_iter=1000):
    """Select n features with sparse L1-penalized Cox model

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
    p_low: float
        Minimum l1 penalizer value
    p_high: float
        Maximum l1 penalizer value
    max_iter: int
        Maximum number of iterations before non-convergence error
    Returns
    -------
    list
        List of n features with non-zero coefficients
    """

    ann = ann[['Event', 'Time to event']]

    def select_features_from_model(model):
        non_zero_coef = np.abs(model.coefs) >= 1e-5
        return df.columns.to_numpy()[non_zero_coef].tolist()
    
    model = CoxRegression(l1_ratio=1, penalizer=p_high)
    model.fit(df, ann)

    p = None
    for i in range(max_iter):
        p_mid = (p_low + p_high) / 2
        model = CoxRegression(l1_ratio=1, penalizer=p_mid)
        model.fit(df, ann)
        n_mid = len(select_features_from_model(model))
        if n_mid == n:
            p = p_mid
            break
        elif n < n_mid:
            p_low = p_mid
        else:
            p_high = p_mid

    if p is None:
        raise Exception(f"Binary search failed to converge to n = {n} features")

    return select_features_from_model(model)
