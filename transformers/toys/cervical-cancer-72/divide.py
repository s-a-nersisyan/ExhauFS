import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = 'data/toys/cervical-cancer-72'

ann = pd.read_csv(f'{BASE_DIR}/annotation.csv', index_col=0)

train, test = train_test_split(ann, test_size=0.5, stratify=ann['Class'])
# train, filter = train_test_split(train_filter, test_size=0.0, stratify=train_filter['Class'])

train['Dataset'] = 'Training'
train['Dataset type'] = 'Training'
# filter['Dataset'] = 'Filtration'
# filter['Dataset type'] = 'Filtration'
test['Dataset'] = 'Validation'
test['Dataset type'] = 'Validation'

combined = pd.concat([train, test], axis=0).sort_index()
# combined = pd.concat([train, filter, test], axis=0).sort_index()

combined.to_csv(f'{BASE_DIR}/[1,1]_annotation.csv')
