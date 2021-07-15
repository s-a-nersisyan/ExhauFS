import numpy as np
from sksurv.compare import compare_survival as logrank_test

from src.core.regression.utils import structure_y_to_sksurv


def logrank(y_true, x, model_coefs):
    """Logrank test
    K-sample log-rank hypothesis test of identical survival functions.
    Compares the pooled hazard rate with each group-specific hazard rate.
    The alternative hypothesis is that the hazard rate of at least one group differs from the others at some time.
    https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.compare.compare_survival.html
    Parameters
    ----------
    y_true :  pandas.DataFrame
        DataFrame with annotation of samples. Two columns are mandatory:
        Event (binary labels), Time to event (float time to event).
    x : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    model_coefs: array-like
        Cox model parameters after fitting
    Returns
    -------
    float
        -log10(logrank test pvalue)
    """
    risk_scores = x.to_numpy().dot(model_coefs.to_numpy())
    group_indicators = risk_scores >= np.median(risk_scores)

    return -np.log10(logrank_test(structure_y_to_sksurv(y_true), group_indicators)[1])
