import pandas as pd

from sksurv.linear_model import CoxPHSurvivalAnalysis

from core.utils import get_datasets


def cox_feature_selection(df, ann, n, datasets=None):
    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann["Dataset"].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples]

    y = ann_subset['Event', 'Time to event']
    columns = df_subset.columns

    scores = []
    model = CoxPHSurvivalAnalysis()
    for j in range(len(columns)):
        df_j = df_subset[:, j:j+1]
        model.fit(df_j, y)
        scores.append(model.score(df_j, y))

    return list(pd.Series(scores, index=columns).sort_values(ascending=False).index)[:n]
