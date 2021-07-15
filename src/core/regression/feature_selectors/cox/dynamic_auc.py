from sksurv.metrics import cumulative_dynamic_auc

from src.core.regression.models import CoxRegression
from src.core.regression.utils import structure_y_to_sksurv
from src.core.utils import get_datasets


def cox_dynamic_auc(df, ann, n, datasets=None, year=3):
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
    datasets : array-like
        List of dataset identifiers which should be used to calculate
        test statistic. By default (None), union of all non-validation
        datasets will be used.
    year: float
        Timepoint for which to calculate AUC score
    Returns
    -------
    list
        List of n features associated with the highest auc.
    """
    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann['Dataset'].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples, ['Event', 'Time to event']]
    structured_y = structure_y_to_sksurv(ann_subset)
    columns = df_subset.columns

    scores = []
    for j, column in enumerate(columns):
        df_j = df_subset[[column]]
        model = CoxRegression()
        model.fit(df_j, ann_subset)
        preds = model.predict(df_j)
        auc, _ = cumulative_dynamic_auc(structured_y, structured_y, preds, [year * 365])
        score = auc[0]

        scores.append(score)

    scores, features = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))

    return features[:n]
