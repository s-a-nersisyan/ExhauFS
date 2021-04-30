import numpy as np


def get_datasets(ann, datasets=None):
    if not datasets:
        datasets = np.unique(ann.loc[ann["Dataset type"] != "Validation", "Dataset"])

    return datasets
