import numpy as np
from lifelines import KaplanMeierFitter


def structure_y_to_sksurv(y):
    return np.array(
        [(bool(a[0]), a[1]) for a in y[['Event', 'Time to event']].to_numpy()],
        dtype=[('event', '?'), ('time', '<f8')],
    )


def plot_kaplan_mayer(y, label):
    kmf = KaplanMeierFitter()
    kmf.fit(y['Time to event'], y['Event'])
    kmf.plot(show_censors=True, label=label, ci_show=False)
