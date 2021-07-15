import pandas as pd


BASE_DIR = 'data/toys/cervical-cancer-72'

data = pd.read_csv(f'{BASE_DIR}/cervical-cancer-72.csv')
data = data.rename(columns={'ca_cervix': 'Class'})

ann = data['Class']
data = data.drop(labels='Class', axis=1)

# неотфильтрованные данные
print(data)
print(ann)
data.to_csv(f'{BASE_DIR}/data.csv')
ann.to_csv(f'{BASE_DIR}/annotation.csv')
