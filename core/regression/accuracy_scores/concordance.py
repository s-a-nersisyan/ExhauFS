import numpy as np

from lifelines.utils import concordance_index as lifelines_concordance_index
from sksurv.metrics import concordance_index_ipcw


def concordance_index(y_true, y_pred):
    return lifelines_concordance_index(
        event_times=y_true['Time to event'],
        predicted_scores=-y_pred,
        event_observed=y_true['Event'],
    )


def concordance_ipcw(y_train, y_test, y_pred):
    structured_y_train = np.array(
        [(bool(a[0]), a[1]) for a in y_train[['Event', 'Time to event']].to_numpy()],
        dtype=[('event', '?'), ('time', '<f8')],
    )
    structured_y_test = np.array(
        [(bool(a[0]), a[1]) for a in y_test[['Event', 'Time to event']].to_numpy()],
        dtype=[('event', '?'), ('time', '<f8')],
    )

    return concordance_index_ipcw(
        structured_y_train,
        structured_y_test,
        y_pred,
    )[0]
