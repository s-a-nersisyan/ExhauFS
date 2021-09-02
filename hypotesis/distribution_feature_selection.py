import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from src.core.regression.accuracy_scores import concordance_index
from src.core.regression.models import CoxRegression


sorted_features = """
GABPB1
ITGA8
BRD4
KDM4B
ZBTB40
LMO2
NR1H2
FXR2
ITGA5
HEXIM1
TBX2
MED26
DMAP1
ZNF770
COL9A3
MLX
OGG1
FN1
ELF1
CHD4
NFIC
LAMB1
GLI1
CNOT3
ZNF639
MAFK
BAP1
MEF2A
hsa-miR-3607-3p|0
MYOCD
SNAI1
GATA3
hsa-miR-200b-3p|0
CD47
CUL4A
PCBP1
POU5F1
GATAD1
ZNF263
HMGXB4
ETV4
BRD3
HNF4A
FOXP1
SIX5
EPAS1
GATAD2A
ZNF574
CEBPA
SLC30A9
NR2F1
PROX1
ZBTB11
RXRA
ITGB4
CASZ1
AGO1
ZBED1
PTBP1
TCF12
MIER2
FOXP3
RB1
ZNF76
FANCL
ASXL1
SOX5
BHLHE40
MEIS2
GREB1
POU2F2
KDM5C
ESRRA
SUPT5H
""".split('\n')[1:-1]

bins = 10
df = pd.read_csv('/Users/vnovosad/Documents/work/Stepa/CD44/data_for_analysis/tcga/expr_isom_preselected.csv', index_col=0)
ann = pd.read_csv('/Users/vnovosad/Documents/work/Stepa/CD44/data_for_analysis/tcga/annotation.csv', index_col=0)
df = df[df.index.isin(ann.index)]
ann = ann[ann.index.isin(df.index)]

samples = ann.loc[ann['Dataset type'].isin(['Training'])].index
df = df.loc[samples]
ann = ann.loc[samples][['Event', 'Time to event']]

model = CoxRegression()
scores = []
for i, feature in enumerate(df.columns):
# for i, feature in enumerate(sorted_features):
    print(feature, i, len(df.columns))
    feature_scores = []
    for _ in range(20):
        # if _ % 100 == 0:
        #     print(_)
        x, _, y, _ = train_test_split(df, ann, train_size=0.5, stratify=ann['Event'])

        model.fit(x[[feature]], y)
        score = model.concordance_index_
        feature_scores.append(score)

    scores.append({'feature': feature, 'score': np.median(feature_scores)})


scores = pd.DataFrame(scores).sort_values(by='score', ascending=False).set_index('feature')
scores.to_csv('features.csv')
print(scores)
#
#
# fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
#
# axs[0].hist(scores['train'], bins=bins, label='train')
# axs[0].set_title('train')
# axs[0].set_xlim([0, 1])
#
# axs[1].hist(scores['filter'], bins=bins, label='filter')
# axs[1].set_title('filter')
# axs[1].set_xlim([0, 1])
#
# axs[2].hist(scores['validation'], bins=bins, label='validation')
# axs[2].set_title('validation')
# axs[2].set_xlim([0, 1])
#
# plt.show()
