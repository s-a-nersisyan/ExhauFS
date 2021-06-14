import numpy as np


def structure_y_to_sksurv(y):
    return np.array(
        [(bool(a[0]), a[1]) for a in y[['Event', 'Time to event']].to_numpy()],
        dtype=[('event', '?'), ('time', '<f8')],
    )
