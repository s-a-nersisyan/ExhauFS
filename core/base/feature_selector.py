class FeatureSelector:
    def __init__(self, df, ann, selector_function, kwargs):
        self.df = df
        self.ann = ann

        self.feature_selector = selector_function
        self.feature_selector_kwargs = kwargs

    def select_features(self, n):
        """Get selected by passed function features.

        Returns
        -------
        list
            List of selected features.
        """
        if self.feature_selector:
            return self.feature_selector(
                self.df,
                self.ann,
                n,
                **self.feature_selector_kwargs,
            )
        else:
            return self.df.columns.to_list()
