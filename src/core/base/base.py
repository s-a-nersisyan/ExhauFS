import pandas as pd
import random
import numpy as np

from multiprocessing import Pool
import time
import math
import itertools

from scipy.stats import binom as binom_stat, rankdata
from scipy.special import binom

from src.core.regression.accuracy_scores import \
    hazard_ratio, \
    dynamic_auc, \
    logrank
from src.core.utils import seconds_to_hours
from .feature_pre_selector import FeaturePreSelector
from .feature_selector import FeatureSelector
from .preprocessor import Preprocessor
from .model import Model


class ExhaustiveBase(
    FeaturePreSelector,
    FeatureSelector,
    Preprocessor,
    Model,
):
    y_features = None

    def __init__(
            self,
            df, ann, n_k, output_dir,
            feature_pre_selector, feature_pre_selector_kwargs,
            feature_selector, feature_selector_kwargs,
            preprocessor, preprocessor_kwargs,
            model, model_kwargs,
            model_cv_ranges, model_cv_folds,
            scoring_functions, main_scoring_function, main_scoring_threshold,
            limit_feature_subsets=False, n_feature_subsets=None,
            shuffle_feature_subsets=True,
            n_processes=1, random_state=None, verbose=True,
    ):
        """Class constructor

         Parameters
         ----------
         df : pandas.DataFrame
             A pandas DataFrame whose rows represent samples
             and columns represent features.
         ann : pandas.DataFrame
             DataFrame with annotation of samples. Three columns are mandatory:
             Class (binary labels), Dataset (dataset identifiers) and
             Dataset type (Training, Filtration, Validation).
         n_k : pandas.DataFrame
             DataFrame with columns n and k defining a grid
             for exhaustive feature selection: n is a number
             of selected features, k is a length of each
             features subset.
         output_dir : str
             Path to dir for output files
         feature_pre_selector : callable
             Function for feature pre-selection. For examples, see
             feature_pre_selectors.py.
         feature_pre_selector_kwargs : dict
             Dict of keyword arguments for feature pre-selector.
         feature_selector : callable
             Function for feature selection. For examples, see
             feature_selectors.py.
         feature_selector_kwargs : dict
             Dict of keyword arguments for feature selector.
         preprocessor : sklearn.preprocessing-like
             Class for data preprocessing, should have fit and
             transform methods. Any method from sklearn.preprocessing
             will be suitable.
         preprocessor_kwargs : dict
             Dict of keyword arguments for preprocessor initialization.
         model : sklearn-like
             Model class, should have fit and
             predict methods. Most of the sklearn models will be suitable.
         model_kwargs : dict
             Dict of keyword arguments for model initialization.
         model_cv_ranges : dict
             Dict defining model parameters which should be
             cross-validated. Keys are parameter names, values are
             iterables for grid search.
         model_cv_folds : int
             Number of fold for K-Folds cross-validation.
         limit_feature_subsets : bool
             If true, limit the number of processed feature subsets.
         n_feature_subsets : int
             Number of processed feature subsets.
         shuffle_feature_subsets : bool
             If true, processed feature subsets are selected randomly.
         max_n : int
             Maximal number of selected features.
         max_estimated_time : float
             Maximal estimated time of pipeline running
         scoring_functions : dict
             Dict with scoring functions which will be calculated
             for each model. Keys are names (arbitrary strings),
             values are sklearn.metrics-like callables (should accept
             y_true, y_pred arguments and return a score).
         main_scoring_function : str
             Key from scoring_functions dict defining the "main" scoring
             function which will be optimized during cross-validation
             and will be used for model filtering.
         main_scoring_threshold : float
             A number defining threshold for model filtering:
             models with score below this threshold on
             training/filtration sets will not be further evaluated.
         n_processes : int
             Number of processes.
         random_state : int
             Random seed (set to an arbitrary integer for reproducibility).
         verbose : bool
             If True, print running time for each pair of n, k.
         """

        FeaturePreSelector.__init__(
            self,
            df, ann,
            preselector_function=feature_pre_selector, kwargs=feature_pre_selector_kwargs,
        )
        Preprocessor.__init__(
            self,
            preprocessor_model=preprocessor, kwargs=preprocessor_kwargs,
        )
        FeatureSelector.__init__(
            self,
            self.df, self.ann,
            output_dir,
            selector_function=feature_selector, kwargs=feature_selector_kwargs,
        )
        Model.__init__(
            self,
            model=model, kwargs=model_kwargs, random_state=random_state,
        )

        self.model_cv_ranges = model_cv_ranges
        self.model_cv_folds = model_cv_folds

        self.n_k = n_k
        self.output_dir = output_dir

        self.n_processes = n_processes
        self.random_state = random_state
        self.verbose = verbose

        self.limit_feature_subsets = limit_feature_subsets
        self.n_feature_subsets = n_feature_subsets
        self.shuffle_feature_subsets = shuffle_feature_subsets

        self.scoring_functions = scoring_functions
        self.main_scoring_function = main_scoring_function
        self.main_scoring_threshold = main_scoring_threshold

        self.datasets_ids = self.ann[['Dataset', 'Dataset type']].drop_duplicates().to_numpy()

    def exhaustive_run(self):
        """Run the pipeline for classifier construction
        using exhaustive feature selection.

        Returns
        -------
        pandas.DataFrame
            DataFrame with constructed classifiers and their
            quality scores.
        """
        # Iterate over n, k pairs
        all_result_dfs = []
        all_counts = []
        summary_n_k = pd.DataFrame(columns=[
            'n', 'k',
            'num_training_reliable',
            'num_validation_reliable',
            'percentage_reliable',
        ])
        for n, k in zip(self.n_k['n'], self.n_k['k']):
            df_n_k_results, _ = self.exhaustive_run_n_k(n, k)
            df_n_k_results['n'] = n
            df_n_k_results['k'] = k

            df_n_k_results.sort_values(
                by=[column for column in df_n_k_results.columns if 'Training' in column],
                ascending=False,
                inplace=True,
            )
            all_result_dfs.append(df_n_k_results)

            # Save models
            res = pd.concat(all_result_dfs, axis=0)
            res.index.name = 'features'
            res['n'] = res['n'].astype(int)
            res['k'] = res['k'].astype(int)
            res.to_csv('{}/models.csv'.format(self.output_dir))

            all_genes = [g for clf in df_n_k_results.index for g in clf.split(';')]
            features_summary = pd.DataFrame(
                [
                    [g, all_genes.count(g)]
                    for g in sorted(list(set(all_genes)))
                ],
                columns=['Gene', 'Count'],
            )
            features_summary['Total'] = len(df_n_k_results)
            features_summary['Percentage'] = features_summary['Count'] / features_summary['Total'] * 100

            # Under null hypothesis, each count is a RV ~ Bin(len(chunk), k/n)
            features_summary['p-value'] = [binom_stat(len(df_n_k_results), k / n).sf(c) for c in features_summary['Count']]
            features_summary['FDR'] = features_summary['p-value'] * len(features_summary) / \
                                      rankdata(features_summary['p-value'])
            features_summary['FDR'] = np.minimum(features_summary['FDR'], 1)
            features_summary = features_summary.sort_values('FDR')
            
            features_summary['n'] = n
            features_summary['k'] = k
            features_summary = features_summary[['n', 'k', 'Gene', 'Count', 'Total', 'Percentage', 'p-value', 'FDR']]

            all_counts.append(features_summary)
            features_summary = pd.concat(all_counts)
            features_summary.to_csv('{}/summary_features.csv'.format(self.output_dir), index=None)

            # Summary table #1: number of models which passed
            # scoring threshold on training + filtration sets,
            # and training + filtration + validation sets

            # All models already passed filtration on training and filtration datasets
            tf_num = len(df_n_k_results)

            # Now do filtration on training, filtration and
            # validation datasets (i.e. all datasets)
            query_string = ' & '.join([
                '(`{};{};{}` >= {})'.format(
                    ds,
                    ds_type,
                    self.main_scoring_function,
                    self.main_scoring_threshold
                ) for ds, ds_type in self.datasets_ids
            ])
            all_num = len(df_n_k_results.query(query_string))

            summary_n_k = summary_n_k.append({
                'n': n, 'k': k,
                'num_training_reliable': tf_num,
                'num_validation_reliable': all_num,
                'percentage_reliable': all_num / tf_num * 100 if tf_num != 0 else 0
            }, ignore_index=True)

            summary_n_k['n'] = summary_n_k['n'].astype(int)
            summary_n_k['k'] = summary_n_k['k'].astype(int)
            summary_n_k['num_training_reliable'] = summary_n_k['num_training_reliable'].astype(int)
            summary_n_k['num_validation_reliable'] = summary_n_k['num_validation_reliable'].astype(int)
            summary_n_k.to_csv('{}/summary_n_k.csv'.format(self.output_dir), index=None)

        return res

    def get_feature_subsets(self, n, k):
        # Do feature selection
        features = self.select_features(n)

        if self.limit_feature_subsets:
            # Generate random subsets, fix for large C_{n}^{k}
            feature_subsets = [random.sample(features, k) for _ in range(self.n_feature_subsets)]
        else:
            # Split feature subsets to chunks for multiprocessing
            feature_subsets = list(itertools.combinations(features, k))

        return feature_subsets

    def get_process_args(self, feature_subsets):
        chunk_size = math.ceil(len(feature_subsets) / self.n_processes)
        process_args = []
        for i in range(self.n_processes):
            start = chunk_size * i
            end = chunk_size * (i + 1) if i < self.n_processes - 1 else len(feature_subsets)
            process_args.append(feature_subsets[start:end])

        return process_args

    def exhaustive_run_n_k(self, n, k):
        """Run the pipeline for classifier construction
        using exhaustive feature selection over number of
        selected features, length of features subsets and
        pre-selected features.

        Parameters
        ----------
        n : int
            Number of selected features.
        k : int
            Length of features subsets.

        Returns
        -------
        pandas.DataFrame, float
            DataFrame with constructed classifiers and their
            quality scores, spent time in hours.
        """

        feature_subsets = self.get_feature_subsets(n, k)

        if self.n_processes > 1:
            # Run exhaustive search in multiple processes
            process_args = self.get_process_args(feature_subsets)

            with Pool(self.n_processes) as p:
                process_results = p.map(self.exhaustive_run_over_chunk,
                                        process_args,
                                        chunksize=1)
        else:
            process_results = [self.exhaustive_run_over_chunk(feature_subsets)]

        # Unpack processes results
        df_results, spent_times = zip(*process_results)

        # Spent time of exhaustive search is max spent time of all processes
        spent_time = max(spent_times)

        if self.verbose:
            main_info = f'Pipeline iteration finished in {spent_time} seconds for n={n}, k={k}'
            tail_infos = [f'n_processes = {self.n_processes}']
            if self.limit_feature_subsets:
                tail_infos.append(f'n_feature_subsets = {self.n_feature_subsets}')
            tail_info = ', '.join(tail_infos)
            print(f'{main_info} ({tail_info})')

        # Merge results
        df_n_k_results = pd.concat(df_results, axis=0)

        if self.limit_feature_subsets and self.shuffle_feature_subsets:
            df_n_k_results.sort_index()

        return df_n_k_results, seconds_to_hours(spent_time)

    def exhaustive_run_over_chunk(self, feature_subsets):
        """Run the pipeline for classifier construction
        using exhaustive feature selection over chunk of
        feature subsets

        Parameters
        ----------
        feature_subsets : list
            list of list of features

        Returns
        -------
        pandas.DataFrame, float
            DataFrame with constructed classifiers and their
            quality scores, spent time in seconds.
        """

        # Fix the start time of process
        start_time = time.time()

        results = []
        for features_subset in feature_subsets:
            features_subset = list(features_subset)

            try:
                model, best_params = self.fit_model(features_subset)
                scores, filtration_passed = self.evaluate_model(model, features_subset)

                item = {
                    'Features subset': features_subset,
                    'Best parameters': best_params,
                    'Scores': scores,
                }
                if filtration_passed:
                    results.append(item)
            except Exception:
                import traceback
                traceback.print_exc()
                print('Excepted ', features_subset)

        score_cols = [
            '{};{};{}'.format(dataset, dataset_type, s)
            for dataset, dataset_type in self.datasets_ids
            for s in self.scoring_functions
        ]

        # list of cv parameters
        parameter_cols = list(self.model_cv_ranges)

        df_results = pd.DataFrame(columns=score_cols + parameter_cols)
        for item in results:
            index = ';'.join(item['Features subset'])
            for dataset_id in item['Scores']:
                for s in item['Scores'][dataset_id]:
                    df_results.loc[index, '{};{}'.format(dataset_id, s)] = item['Scores'][dataset_id][s]

            for parameter in parameter_cols:
                df_results.loc[index, parameter] = item['Best parameters'][parameter]

        # Calculate spent time of process
        spent_time = time.time() - start_time

        return df_results, spent_time

    def fit_model(self, features_subset):
        """Fit classifier given features subset

        Parameters
        ----------
        features_subset : list
            list of features which should be used for
            classifier fitting.

        Returns
        -------
        sklearn.model-like, model best parameters, sklearn.preprocessing-like
            Model and parameters and preprocessor fitted on the
            training set.
        """

        # Extract training set
        X_train = self.df.loc[self.ann['Dataset type'] == 'Training', features_subset]
        y_train = self.ann.loc[self.ann['Dataset type'] == 'Training', self.y_features]
        if self.check_if_model_needs_numpy():
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()

        X_train = self.preprocess(X_train, is_fit=True)

        model, best_params = self.get_best_cv_model(
            X_train,
            y_train,
            scoring_functions=self.scoring_functions,
            main_scoring_function=self.main_scoring_function,
            cv_ranges=self.model_cv_ranges,
            cv_folds=self.model_cv_folds,
        )

        model.fit(X_train, y_train)

        return model, best_params

    def evaluate_model(self, model, features_subset):
        """Evaluate classifier given features subset

        Parameters
        ----------
        model : sklearn.model-like
            Fitted model object with a method predict(X).
        features_subset : list
            list of features which should be used for
            classifier evaluation.

        Returns
        -------
        dict, bool
            Dict with scores for each dataset and
            boolean value indicating whether a
            model passed given threshold on
            training and filtration sets.
        """

        scores = {}
        filtration_passed = True
        for dataset, dataset_type in self.datasets_ids:
            dataset_id = f'{dataset};{dataset_type}'
            X_test = self.df.loc[
                (self.ann['Dataset'] == dataset) & (self.ann['Dataset type'] == dataset_type),
                features_subset,
            ]
            y_test = self.ann.loc[
                (self.ann['Dataset'] == dataset) & (self.ann['Dataset type'] == dataset_type),
                self.y_features,
            ]
            if self.check_if_model_needs_numpy():
                X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

            X_test = self.preprocess(X_test)

            # Make predictions
            y_pred = model.predict(X_test)

            scores[dataset_id] = {}
            for s in self.scoring_functions:
                if self.check_if_method_needs_proba(s):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    score = self.scoring_functions[s](y_test, y_proba)
                else:
                    if self.scoring_functions[s] in [hazard_ratio, logrank]:
                        score = self.scoring_functions[s](y_test, X_test, model.coefs)
                    elif self.scoring_functions[s] in [dynamic_auc]:
                        score = self.scoring_functions[s](
                            self.ann.loc[self.ann['Dataset type'] == 'Training', self.y_features],
                            y_test,
                            y_pred,
                        )
                    else:
                        score = self.scoring_functions[s](y_test, y_pred)

                scores[dataset_id][s] = score

            if (
                    dataset_type in ['Training', 'Filtration']
                    and scores[dataset_id][self.main_scoring_function] < self.main_scoring_threshold
            ):
                filtration_passed = False

        return scores, filtration_passed

    def estimate_run_n_k_time(self, n, k, time_per_iteration):
        """Estimate run time of the pipeline for classifiers
        construction using exhaustive feature selection over
        number of selected features and length of features subsets.

        Parameters
        ----------
        n : int
            Number of selected features.
        k : int
            Length of features subsets.
        time_per_iteration : float
            Time spent on running of the pipeline over n selected features
            and k features subsets using only restricted number of processed
            feature subsets (given by self.n_feature_subsets).

        Returns
        -------
        float
            Estimated time that the pipeline will spend on running over
            n selected features and k features subsets without any restriction
            on the number of processed feature subsets.
        """
        if self.limit_feature_subsets:
            coef = binom(n, k)
            if coef > self.n_feature_subsets:
                return coef * time_per_iteration / self.n_feature_subsets

        return time_per_iteration
