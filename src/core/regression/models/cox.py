import pandas as pd
import warnings

from lifelines import CoxPHFitter

warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*Ill-conditioned matrix.*')


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
