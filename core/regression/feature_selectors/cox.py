import time
import numpy as np

from scipy.stats import spearmanr

from core.regression.accuracy_scores import hazard_ratio
from core.regression.models import CoxRegression
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc

from core.utils import get_datasets


def cox_feature_selection(df, ann, n, datasets=None, method=''):
    datasets = get_datasets(ann, datasets)

    samples = ann.loc[ann['Dataset'].isin(datasets)].index
    df_subset = df.loc[samples]
    ann_subset = ann.loc[samples, ['Event', 'Time to event']]
    y = ann_subset['Time to event'].to_numpy()
    structured_y = np.array(
        [(bool(a[0]), a[1]) for a in ann_subset[['Event', 'Time to event']].to_numpy()],
        dtype=[('event', '?'), ('time', '<f8')],
    )
    columns = df_subset.columns

    scores = []
    start = time.time()
    for j, column in enumerate(columns):
        if j and j % 100 == 0:
            print(time.time() - start, j, len(columns))
        if method == 'concordance':
            df_j = df_subset[[column]]
            model = CoxRegression()
            model.fit(df_j, ann_subset)
            score = model.concordance_index_

        if method == 'likelihood':
            df_j = df_subset[[column]]
            model = CoxRegression()
            model.fit(df_j, ann_subset)
            score = model.log_likelihood_

        if method == 'concordance_ipcw':
            df_j = df_subset[[column]]
            model = CoxRegression()
            model.fit(df_j, ann_subset)
            preds = model.predict(df_j)
            score = concordance_index_ipcw(
                structured_y,
                structured_y,
                preds,
            )[0]

        if method == 'auc':
            df_j = df_subset[[column]]
            model = CoxRegression()
            model.fit(df_j, ann_subset)
            preds = model.predict(df_j)
            auc, _ = cumulative_dynamic_auc(structured_y, structured_y, preds, [3*365])
            score = auc[0]

        if method == 'hazard_ratio':
            df_j = df_subset[[column]]
            model = CoxRegression()
            model.fit(df_j, ann_subset)
            score = hazard_ratio(ann_subset, df_j, model.coefs)

        if method == 'correlation':
            df_j = df_subset[column].to_numpy()
            score = 1 - spearmanr(df_j, y).pvalue

        scores.append(score)

    print(f'Took {time.time() - start} seconds for {len(columns)} columns')
    scores, features = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))
    with open('sorted_features.txt', 'w') as f:
        for score, feature in zip(scores, features):
            f.write(f'{feature} {score}\n')

    return features[:n]
