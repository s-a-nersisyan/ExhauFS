import pandas as pd
import time

from core.regression.models import CoxRegression

from core.utils import get_datasets


def cox_feature_selection(df, ann, n, datasets=None):
    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann['Dataset'].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples, ['Event', 'Time to event']]
    columns = df_subset.columns

    scores = []
    model = CoxRegression()
    start = time.time()
    df_subset = pd.concat([df_subset, ann_subset], axis=1)
    for j, column in enumerate(columns):
        if j and j % 100 == 0:
            print(time.time() - start, j, len(columns))
        df_j = df_subset[[column]]
        model.fit(df_j, ann_subset)
        scores.append(model.concordance_index_)

    print(f'Took {time.time() - start} seconds for {len(columns)} columns')
    _, sorted_columns = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))
    with open('sorted_features.txt', 'w') as f:
        for feature in sorted_columns:
            f.write(str(feature) + '\n')

    return sorted_columns[:n]
