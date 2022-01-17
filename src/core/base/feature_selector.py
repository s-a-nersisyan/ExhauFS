from src.core.classification.feature_selectors import l1_logistic_regression
from src.core.regression.feature_selectors import l1_cox


class FeatureSelector:
    def __init__(self, df, ann, output_dir, selector_function, kwargs):
        self.df = df
        self.ann = ann

        self.output_dir = output_dir

        self.feature_selector = selector_function
        self.feature_selector_kwargs = kwargs

        self.set_sorted_features()
        if self.sorted_features:
            self.save_sorted_features()

    def set_sorted_features(self):
        self.sorted_features = None
        if self.feature_selector:
            if self.feature_selector not in [l1_logistic_regression, l1_cox]:
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

        if self.sorted_features:
            return self.sorted_features[:n]

        return self.feature_selector(self.df, self.ann, n=n, **self.feature_selector_kwargs)
