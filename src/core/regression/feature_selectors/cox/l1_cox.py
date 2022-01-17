import numpy as np

from src.core.regression.models import CoxRegression
from src.core.wrappers import feature_selector_wrapper


@feature_selector_wrapper()
def l1_cox(df, ann, n):
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
    Returns
    -------
    list
        List of n features with non-zero coefficients
    """

    ann = ann[['Event', 'Time to event']]
    columns = df.columns
    
    def select_features_from_model(model):
        non_zero_coef = np.abs(model.coefs) >= 1e-5
        return df.columns.to_numpy()[non_zero_coef].tolist()
    
    p_low, p_high = 0, 1e+6
    model = CoxRegression(l1_ratio=1, penalizer=p_high)
    model.fit(df, ann)
    n_low, n_high = len(df.columns), len(select_features_from_model(model))

    max_iter = 1000
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
            n_low = n_mid
        else:
            p_high = p_mid
            n_high = n_mid

    if p is None:
        raise Exception(f"Binary search failed to converge to n = {n} features")

    return select_features_from_model(model)
