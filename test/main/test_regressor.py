import os
import random
import unittest
import pandas as pd

from src.core import accuracy_scores, feature_selectors
from src.core.preprocessors import *
from src.core.regression.regressors import *
from src.core.regression.regression import ExhaustiveRegression

random.seed(0)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = f'{BASE_DIR}/tmp'


class TestRegressor(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100
        self.data = pd.DataFrame.from_dict({
            f'feature_{feature_index}': [random.random() for _ in range(self.n_samples)]
            for feature_index in range(10)
        })
        self.ann = pd.DataFrame.from_dict({
            'Event': [random.randint(0, 1) for _ in range(self.n_samples)],
            'Time to event': [(i + 1) * 10000 / self.n_samples for i in range(self.n_samples)],
            'Dataset': 'Testing',
            'Dataset type': [
                'Validation' if i % 3 == 0 else 'Training' if i % 2 == 0 else 'Filtration'
                for i in range(self.n_samples)
            ],
        })

        self.n_k_grid = pd.DataFrame([
            {'n': 5, 'k': 2},
            {'n': 10, 'k': 9},
        ])

        self.model = ExhaustiveRegression(
            df=self.data,
            ann=self.ann,
            n_k=self.n_k_grid,
            output_dir=TMP_DIR,
            feature_pre_selector=None,
            feature_pre_selector_kwargs={},
            feature_selector=feature_selectors.cox_concordance,
            feature_selector_kwargs={},
            preprocessor=None,
            preprocessor_kwargs={},
            model=CoxRegression,
            model_kwargs={},
            model_cv_ranges={},
            model_cv_folds=0,
            scoring_functions={
                s: getattr(accuracy_scores, s)
                for s in ["concordance_index", "dynamic_auc", "hazard_ratio", "logrank"]
            },
            main_scoring_function='concordance_index',
            main_scoring_threshold=0.3,
            random_state=0,
        )

    def test_run(self):
        lhs = self.model.exhaustive_run().astype(float).round(10)

        rhs = pd.read_csv(f'{BASE_DIR}/regressor_result.test', index_col=0).round(10)

        self.assertTrue(lhs.equals(rhs))


if __name__ == '__main__':
    if not os.path.isdir(TMP_DIR):
        os.makedirs(TMP_DIR)
    unittest.main()
