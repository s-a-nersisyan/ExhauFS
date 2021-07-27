from sksurv.metrics import cumulative_dynamic_auc

from src.core.regression.utils import structure_y_to_sksurv


def dynamic_auc(y_train, y_test, y_pred, year=3):
    """Dynamic or Time-Dependent AUC
     is the average of how often a model says X is greater than Y when,
     in the observed data, X is indeed greater than Y
    https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index

    Parameters
    ----------
    y_true :  pandas.DataFrame
        DataFrame with annotation of samples. Two columns are mandatory:
        Event (binary labels), Time to event (float time to event).
    y_test :  pandas.DataFrame
        DataFrame with annotation of samples. Two columns are mandatory:
        Event (binary labels), Time to event (float time to event).
    y_pred : array-like
        List of predicted risk scores.
    year: float
        Timepoint at which to calculate the AUC score
    Returns
    -------
    float [0, 1]
        dynamic auc for specified year
    """
    structured_y_train = structure_y_to_sksurv(y_train)
    structured_y_test = structure_y_to_sksurv(y_test)

    return cumulative_dynamic_auc(
        structured_y_train,
        structured_y_test,
        y_pred,
        [year],
    )[0][0]
