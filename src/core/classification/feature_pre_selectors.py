from scipy.stats import f_oneway


def f_test(df, ann):
    """Pre-select features without difference between types of dataset

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame whose rows represent samples
        and columns represent features.
    ann : pandas.DataFrame
        DataFrame with annotation of samples. This argument is
        actually not used by the function.
    Returns
    -------
    list
        List of features without difference between types of dataset intersected with a list of
        features from a given DataFrame.
    """
    dataset_ids = ann[['Dataset', 'Dataset type']].drop_duplicates().to_numpy()

    samples = [
        df.loc[(ann['Dataset'] == dataset) & (ann['Dataset type'] == dataset_type)]
        for dataset, dataset_type in dataset_ids
    ]
    statistics, pvalues = f_oneway(*samples, axis=0)

    return df.columns[pvalues > 0.05].to_list()
