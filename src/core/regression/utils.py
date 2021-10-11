import numpy as np
from lifelines import KaplanMeierFitter


def structure_y_to_sksurv(y):
    return np.array(
        [(bool(a[0]), a[1]) for a in y[['Event', 'Time to event']].to_numpy()],
        dtype=[('event', '?'), ('time', '<f8')],
    )


def plot_kaplan_mayer(y, label, title='A'):
    kmf = KaplanMeierFitter()
    kmf.fit(y['Time to event'], y['Event'])
    ax = kmf.plot(show_censors=True, label=label, ci_show=False)
    ax.set_title(title, loc="left", fontdict={"fontsize": "xx-large", "fontweight": "bold"})

