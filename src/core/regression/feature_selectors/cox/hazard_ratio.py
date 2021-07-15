from src.core.regression.accuracy_scores import hazard_ratio
from src.core.regression.models import CoxRegression
from src.core.utils import get_datasets


def cox_hazard_ratio(df, ann, n, datasets=None):
    """Select n features with the highest hazard ratio on one-factor Cox regression.

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
    Returns
    -------
    list
        List of n features associated with the highest hazard ratio.
    """
    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann['Dataset'].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples, ['Event', 'Time to event']]
    columns = df_subset.columns

    scores = []
    for j, column in enumerate(columns):
        df_j = df_subset[[column]]
        model = CoxRegression()
        model.fit(df_j, ann_subset)
        score = hazard_ratio(ann_subset, df_j, model.coefs)

        scores.append(score)

    scores, features = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))

    return features[:n]
