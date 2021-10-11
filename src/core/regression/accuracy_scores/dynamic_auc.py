import numpy as np

from sksurv.metrics import cumulative_dynamic_auc, _check_estimate_2d
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import check_y_survival

from src.core.regression.utils import structure_y_to_sksurv


def dynamic_fpr_tpr(y_train, y_test, y_pred, year=3):
    survival_train = structure_y_to_sksurv(y_train)
    survival_test = structure_y_to_sksurv(y_test)

    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(y_pred, test_time, [year])

    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))

    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)

    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)

    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls

    return false_pos, true_pos


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
