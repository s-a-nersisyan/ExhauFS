import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from src.core.regression.accuracy_scores import concordance_index
from src.core.regression.models import CoxRegression


bins = 10
feature = 'GABPB1'
df = pd.read_csv('/Users/vnovosad/Documents/work/Stepa/CD44/data_for_analysis/tcga/expr_isom_preselected.csv', index_col=0)
ann = pd.read_csv('/Users/vnovosad/Documents/work/Stepa/CD44/data_for_analysis/tcga/annotation.csv', index_col=0)
df = df[df.index.isin(ann.index)]
ann = ann[ann.index.isin(df.index)]

df = df[[feature]]
ann = ann[['Event', 'Time to event']]

model = CoxRegression()
scores = []
for _ in range(10000):
    if _ % 100 == 0:
        print(_)
    train_x, filter_val_x, train_y, filter_val_y = train_test_split(df, ann, train_size=0.33, stratify=ann['Event'])
    filter_x, val_x, filter_y, val_y = train_test_split(filter_val_x, filter_val_y, train_size=0.5, stratify=filter_val_y['Event'])

    model.fit(train_x, train_y)
    score_train = model.concordance_index_

    score_filter = concordance_index(filter_y, model.predict(filter_x))
    score_val = concordance_index(val_y, model.predict(val_x))

    scores.append({'train': score_train, 'filter': score_filter, 'validation': score_val})

scores = pd.DataFrame(scores)


fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)

axs[0].hist(scores['train'], bins=bins, label='train')
axs[0].set_title('train')
axs[0].set_xlim([0, 1])

axs[1].hist(scores['filter'], bins=bins, label='filter')
axs[1].set_title('filter')
axs[1].set_xlim([0, 1])

axs[2].hist(scores['validation'], bins=bins, label='validation')
axs[2].set_title('validation')
axs[2].set_xlim([0, 1])

plt.show()
