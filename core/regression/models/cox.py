import pandas as pd

from lifelines import CoxPHFitter


class CoxRegression(CoxPHFitter):
    def fit(self, x, y):
        return super().fit(
            pd.concat([x, y], axis=1),
            event_col='Event',
            duration_col='Time to event',
        )

    def predict(self, df):
        return self.predict_partial_hazard(df)

    @property
    def coefs(self):
        return self.params_
