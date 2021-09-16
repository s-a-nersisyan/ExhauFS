from pprint import pprint
import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import scipy.cluster.hierarchy as spc

from src.core.regression.models import CoxRegression


df = pd.read_csv('/home/v-novosad/CD44/data_for_analysis/tcga/new-expr_isom_preselected.csv', index_col=0)
ann = pd.read_csv('/home/v-novosad/CD44/data_for_analysis/tcga/annotation.csv', index_col=0)
df = df[df.index.isin(ann.index)]
ann = ann[ann.index.isin(df.index)]

features = df.columns


def clustering_labels():
    corr = df[features].corr().values

    clustering = DBSCAN(eps=0.2, metric='precomputed')
    clustering = clustering.fit(1 - corr)
    # print(1 - corr)

    labels = clustering.labels_ + 1

    return labels


def correlation_scores(n=20, tresh=0.95):
    columns = df.columns[:n]

    samples = ann.loc[ann['Dataset type'].isin(['Training'])].index
    df_local = df.loc[samples]
    ann_local = ann.loc[samples][['Event', 'Time to event']]

    scores = {}
    for i, column_l in enumerate(columns):
        for j, column_r in enumerate(columns):
            if j > i:
                score = spearmanr(df_local[column_l], df_local[column_r]).correlation

                if score > tresh:
                    if column_l not in scores:
                        scores[column_l] = {}
                    scores[column_l][column_r] = score

    return scores


def concs(n=20):
    columns = df.columns[:n]

    samples = ann.loc[ann['Dataset type'].isin(['Training'])].index
    df_local = df.loc[samples]
    ann_local = ann.loc[samples][['Event', 'Time to event']]

    model = CoxRegression()

    scores = []
    for j, column in enumerate(columns):
        df_j = df_local[[column]]
        model.fit(df_j, ann_local)
        score = model.concordance_index_

        scores.append(score)

    scores, features = zip(*sorted(zip(scores, columns), key=lambda x: x[0], reverse=True))

    print(pd.DataFrame(
        [{'feature': features[i], 'concordance': scores[i]} for i in range(len(scores))]
    ).sort_values(by='concordance', ascending=False))


def cox_scores(n=20, tresh=0.01):
    columns = df.columns[:n]

    samples = ann.loc[ann['Dataset type'].isin(['Training'])].index
    df_local = df.loc[samples]
    ann_local = ann.loc[samples][['Event', 'Time to event']]

    model = CoxRegression()
    scores = {}
    for i, column_l in enumerate(columns):
        print(f'{i}/{len(columns)} {column_l}')
        for j, column_r in enumerate(columns):
            if j > i and f'{column_r};{column_l}' not in scores:
                df_l = df_local[[column_l]]
                df_r = df_local[[column_r]]
                df_both = df_local[[column_l, column_r]]

                model.fit(df_l, ann_local)
                score_l = model.concordance_index_

                model.fit(df_r, ann_local)
                score_r = model.concordance_index_

                model.fit(df_both, ann_local)
                score_both = model.concordance_index_

                corr_score = spearmanr(df_local[column_l], df_local[column_r]).correlation

                if (abs(score_both - score_l) < tresh and abs(score_both - score_r) < tresh) or corr_score > 0.95:
                        scores[f'{column_l};{column_r}'] = {
                            'concordance_diff': max([abs(score_both - score_l), abs(score_both - score_r)]),
                            'correlation': corr_score,
                            'concordance_left': score_l,
                            'concordance_right': score_r,
                        }

    # return pd.DataFrame(scores).T.sort_values(by='correlation', ascending=False)
    return pd.DataFrame(scores).T.sort_index()


concs()
scores = cox_scores()
print(scores)
scores.to_csv('scores.csv')

# labels = clustering_labels()
# labels_matrix = np.array(labels).reshape(len(labels), 1).dot(np.array(labels).reshape(1, len(labels)))
# print(labels_matrix)
# print(corr * labels_matrix)
# sns.clustermap(corr, mask=1 - labels_matrix)
# plt.show()

# print(list(features[labels.astype(bool)]))
# print(df[features[labels == 0]])
