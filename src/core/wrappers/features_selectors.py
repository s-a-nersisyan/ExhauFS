from functools import wraps


def feature_selector_wrapper():
    def wrapper(func):
        @wraps(func)
        def selector(df, ann, n, *args, use_filtration=False, **kwargs):
            dataset_types = ['Training']
            if use_filtration:
                dataset_types.append('Filtration')

            samples = ann.loc[ann['Dataset type'].isin(dataset_types)].index
            df_subset = df.loc[samples]
            ann_subset = ann.loc[samples]

            return func(df_subset, ann_subset, n, *args, **kwargs)

        return selector

    return wrapper
