from lifelines.utils import concordance_index as lifelines_concordance_index


def concordance_index(y_true, y_pred):
    return lifelines_concordance_index(
        event_times=y_true['Time to event'],
        predicted_scores=-y_pred,
        event_observed=y_true['Event'],
    )
