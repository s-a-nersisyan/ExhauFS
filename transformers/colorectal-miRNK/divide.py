import json
import pandas as pd


BASE_DIR = 'data/colorectal-miRNK'

ann = pd.read_csv(f'{BASE_DIR}/annotation.csv', index_col=0)

# сортировка по евенту и времени для stratification разделения на типы
ann = ann.sort_values(by=['Event', 'Time to event'], ascending=False)

dataset_type_split_ways = [[3, 3], [2, 4]]

for split_way in dataset_type_split_ways:
    dataset_types = [
        'Training' if i % split_way[0] == 0
        else 'Validation' if (i - 1) % split_way[1] == 0
        else 'Filtration'
        for i in range(len(ann.index))
    ]
    ann['Dataset'] = dataset_types
    ann['Dataset type'] = dataset_types

    ann.sort_index().to_csv(f'{BASE_DIR}/annotation_{json.dumps(split_way)}.csv')
    print('Training 0', len(ann[(ann['Dataset type'] == 'Training') & (ann['Event'] == 0)]))
    print('Training 1', len(ann[(ann['Dataset type'] == 'Training') & (ann['Event'] == 1)]))
    print('Validation 0', len(ann[(ann['Dataset type'] == 'Validation') & (ann['Event'] == 0)]))
    print('Validation 1', len(ann[(ann['Dataset type'] == 'Validation') & (ann['Event'] == 1)]))
    print('Filtration 0', len(ann[(ann['Dataset type'] == 'Filtration') & (ann['Event'] == 0)]))
    print('Filtration 1', len(ann[(ann['Dataset type'] == 'Filtration') & (ann['Event'] == 1)]))
