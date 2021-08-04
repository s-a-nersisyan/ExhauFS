from sksurv.metrics import cumulative_dynamic_auc

from src.core.wrappers import feature_selector_wrapper
from src.core.regression.models import CoxRegression
from src.core.regression.utils import structure_y_to_sksurv


@feature_selector_wrapper()
def cox_dynamic_auc(df, ann, n, year=3):
    """Select n features with the highest time-dependent auc on one-factor Cox regression.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. Three columns are mandatory:
        Class (binary labels), Dataset (dataset identifiers) and
        Dataset type (Training, Filtration, Validation).
    n : int
        Number of features to select.
    year: float
        Timepoint for which to calculate AUC score
    Returns
    -------
    list
        List of n features associated with the highest auc.
    """
    ann = ann[['Event', 'Time to event']]

    structured_y = structure_y_to_sksurv(ann)
    columns = df.columns

    scores = []
    for j, column in enumerate(columns):
        df_j = df[[column]]
        model = CoxRegression()
        model.fit(df_j, ann)
        preds = model.predict(df_j)
        auc, _ = cumulative_dynamic_auc(structured_y, structured_y, preds, [year])
        score = auc[0]

        scores.append(score)

    scores, features = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))

    return features[:n]
