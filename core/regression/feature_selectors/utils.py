def save_sorted_features(scores, features, file_path='sorted_features.txt'):
    with open(file_path, 'w') as f:
        for score, feature in zip(scores, features):
            f.write(f'{feature} {score}\n')
