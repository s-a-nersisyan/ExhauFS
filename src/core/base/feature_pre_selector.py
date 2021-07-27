class FeaturePreSelector:
    def __init__(self, df, ann, preselector_function, kwargs):
        self.df = df
        self.ann = ann

        self.feature_pre_selector = preselector_function
        self.feature_pre_selector_kwargs = kwargs

        self.df = self.df[self.pre_selected_features]

    @property
    def pre_selected_features(self):
        """Get pre-selected features.

        Returns
        -------
        list
            List of pre-selected features.
        """
        if self.feature_pre_selector:
            return self.feature_pre_selector(
                self.df,
                self.ann,
                **self.feature_pre_selector_kwargs,
            )
        else:
            return self.df.columns.to_list()
