class FeatureSelector:
    def __init__(self, df, ann, output_dir, selector_function, kwargs):
        self.df = df
        self.ann = ann

        self.output_dir = output_dir

        self.feature_selector = selector_function
        self.feature_selector_kwargs = kwargs

        self.set_sorted_features()
        self.save_sorted_features()

    def set_sorted_features(self):
        if self.feature_selector:
            self.sorted_features = self.feature_selector(
                self.df,
                self.ann,
                n=len(self.df.columns),
                **self.feature_selector_kwargs,
            )
        else:
            self.sorted_features = self.df.columns.to_list()

    def save_sorted_features(self):
        with open('{}/sorted_features.txt'.format(self.output_dir), 'w') as f:
            f.write('\n'.join(self.sorted_features))

    def select_features(self, n):
        """Get selected by passed function features.

        Returns
        -------
        list
            List of selected features.
        """

        return self.sorted_features[:n]
