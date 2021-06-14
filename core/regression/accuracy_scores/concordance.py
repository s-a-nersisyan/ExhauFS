from lifelines.utils import concordance_index as lifelines_concordance_index


def concordance_index(y_true, y_pred):
    """Concordance index
     is the average of how often a model says X is greater than Y when,
     in the observed data, X is indeed greater than Y
    https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index

    Parameters
    ----------
    y_true :  pandas.DataFrame
        DataFrame with annotation of samples. Two columns are mandatory:
        Event (binary labels), Time to event (float time to event).
    y_pred : array-like
        List of predicted risk scores.
    Returns
    -------
    float [0, 1]
        concordance_index
    """
    return lifelines_concordance_index(
        event_times=y_true['Time to event'],
        predicted_scores=-y_pred,
        event_observed=y_true['Event'],
    )
