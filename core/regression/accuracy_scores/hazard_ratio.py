import numpy as np


def hazard_ratio(y_true, x, model_coefs):
    """Hazard ratio can be interpreted as the chance of an event occurring
    in the group A divided by the chance of the event occurring in the group B
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
        hazard_ratio
    """
    risk_scores = x.to_numpy().dot(model_coefs.to_numpy())
    group_indicators = risk_scores >= np.median(risk_scores)
    grouped_y = y_true.copy()
    grouped_y['group'] = group_indicators

    i_a = [len(group_indicators[group_indicators == True])]
    i_b = [len(group_indicators) - i_a[0]]
    o_a = []
    o_b = []
    e_a = []
    e_b = []
    sorted_times = sorted(y_true['Time to event'][y_true['Event'] == 1].unique())

    for event_time in sorted_times:
        groups = group_indicators[y_true['Time to event'] == event_time]

        o_a.append(len(groups[groups == True]))
        o_b.append(len(groups) - o_a[-1])

        total_dead = (o_a[-1] + o_b[-1])
        total_alive = (i_a[-1] + i_b[-1])

        e_a.append(i_a[-1] * total_dead / total_alive)
        e_b.append(i_b[-1] * total_dead / total_alive)

        i_a.append(i_a[-1] - o_a[-1])
        i_b.append(i_b[-1] - o_b[-1])

    return (sum(o_a) / sum(e_a)) / (sum(o_b) / sum(e_b))
