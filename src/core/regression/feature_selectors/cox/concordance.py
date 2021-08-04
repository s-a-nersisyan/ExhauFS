from src.core.regression.models import CoxRegression
from src.core.wrappers import feature_selector_wrapper


@feature_selector_wrapper()
def cox_concordance(df, ann, n):
    """Select n features with the highest concordance index on one-factor Cox regression.

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
    Returns
    -------
    list
        List of n features associated with the highest c-index.
    """
    ann = ann[['Event', 'Time to event']]

    columns = df.columns

    scores = []
    for j, column in enumerate(columns):
        df_j = df[[column]]
        model = CoxRegression()
        model.fit(df_j, ann)
        score = model.concordance_index_

        scores.append(score)

    scores, features = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))

    return features[:n]
