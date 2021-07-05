import os
import random
import unittest
import pandas as pd

from src.core import accuracy_scores, feature_selectors
from src.core.preprocessors import *
from src.core.classification.classifiers import *
from src.core.classification.classification import ExhaustiveClassification

random.seed(0)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = f'{BASE_DIR}/tmp'


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100
        self.data = pd.DataFrame.from_dict({
            f'feature_{feature_index}': [random.random() for _ in range(self.n_samples)]
            for feature_index in range(10)
        })
        self.ann = pd.DataFrame.from_dict({
            'Class': [random.randint(0, 1) for _ in range(self.n_samples)],
            'Dataset': 'Testing',
            'Dataset type': [random.choice(['Training', 'Filtration', 'Validation']) for _ in range(self.n_samples)],
        })

        self.n_k_grid = pd.DataFrame([
            {'n': 5, 'k': 2},
            {'n': 10, 'k': 9},
        ])

        self.model = ExhaustiveClassification(
            df=self.data,
            ann=self.ann,
            n_k=self.n_k_grid,
            output_dir=TMP_DIR,
            feature_pre_selector=None,
            feature_pre_selector_kwargs={},
            feature_selector=feature_selectors.t_test,
            feature_selector_kwargs={},
            preprocessor=KBinsDiscretizer,
            preprocessor_kwargs={
                'n_bins': 2,
                'encode': 'ordinal',
            },
            model=SVC,
            model_kwargs={
                'kernel': 'linear',
                'class_weight': 'balanced',
            },
            model_cv_ranges={
                'C': [0.00390625, 0.015625, 0.0625, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0],
            },
            model_cv_folds=5,
            scoring_functions={s: getattr(accuracy_scores, s) for s in ['TPR', 'TNR', 'min_TPR_TNR']},
            main_scoring_function='min_TPR_TNR',
            main_scoring_threshold=0.3,
            random_state=0,
        )

    def test_run(self):
        lhs = self.model.exhaustive_run().astype(float).round(10)

        rhs = pd.read_csv(f'{BASE_DIR}/classifier_result.test', index_col=0).round(10)

        self.assertTrue(lhs.equals(rhs))


if __name__ == '__main__':
    if not os.path.isdir(TMP_DIR):
        os.makedirs(TMP_DIR)
    unittest.main()
