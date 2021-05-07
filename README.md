# ExhaustiveFS
Exhaustive feature selection for classication and survival analysis.

**TODO:** add table of contents.

# Introduction
The main idea underlying ExhaustiveFS is the exhaustive search of feature subsets for constructing the most powerfull classification and survival regression models. Since computational complexity of such approach grows exponentially with respect to combination length, we first narrow down features list in order to make search practically feasible. Briefly, the following pipeline is implemented:
1. **Feature pre-selection:** select fixed number of features for the next steps.
2. **Feature selection:** select *n* features for exhaustive search.
3. **Exhaustive search:** iterate through all possible *k*-element feature subsets and fit classification/regression models.

Values of *n* and *k* actually define running time of the pipeline (there are *C<sub>n</sub><sup>k</sup>* feature subsets). For example, iterating through all 8-gene signatures composed of *n = 20* genes is possible (see example breast cancer data below), while search for over *n = 1000* genes will never end even on the most powerful supercomputer.

Input data can consist from different batches (datasets), and each dataset should be labeled by one of the following types:
1. **Training set:** samples from training datasets will be used for tuning classification/regression models. At least one such dataset is required; if multiple given, the union will be used.
2. **Filtration set:** all tuned models will be first evaluated on training and filtration sets. If specified thresholds for accuracy are reached, model will be evaluated on validation (test) sets. The use of filtration sets is optional.
3. **Validation (test) set:** performance of models which passed filtration thresholds are then evaluated on validation sets. At least one such dataset is required; if multiple given, model will be evaluated on all test sets independently.

**TODO:** add flowchart.

# Requirements

**TODO:** list of requirements + how to install.

# Running ExhaustiveFS

## Step 1: data preparation

Before running the tool, you should prepare two csv tables containing actual data and its annotation. Both for classification and survival analysis data table should contain numerical values associated with samples (rows) and features (columns):

|            | Feature 1 | Feature 2 |
| ---------- | --------- | --------- |
| Sample 1   | 17.17     | 365.1     |
| Sample 2   | 56.99     | 123.9     |
| ...        |           |           |
| Sample 98  | 22.22     | 123.4     |
| Sample 99  | 23.23     | 567.8     |
| ...        |           |           |
| Sample 511 | 10.82     | 665.8     |
| Sample 512 | 11.11     | 200.2     |

Sample annotation table formats are different for classification and survival analysis. For classification it should contain binary indicator of sample class (e.g., 1 for recurrent tumor and 0 for non-recurrent), dataset (batch) label and dataset type (Training/Filtration/Validation). Note that annotation should be present for each sample listed in the data table in the same order:

|            | Class | Dataset  | Dataset type |
| ---------- | ----- | -------- | ------------ |
| Sample 1   | 1     | GSE3494  | Training     |
| Sample 2   | 0     | GSE3494  | Training     |
| ...        |       |          |              |
| Sample 98  | 0     | GSE12093 | Filtration   |
| Sample 99  | 0     | GSE12093 | Filtration   |
| ...        |       |          |              |
| Sample 511 | 1     | GSE1456  | Validation   |
| Sample 512 | 1     | GSE1456  | Validation   |

For survival analysis, annotation table should contain binary event indicator and time to event:

|            | Event | Time to event | Dataset  | Dataset type |
| ---------- | ----- | ------------- | -------- | ------------ |
| Sample 1   | 1     | 100.1         | GSE3494  | Training     |
| Sample 2   | 0     | 500.2         | GSE3494  | Training     |
| ...        |       |               |          |              |
| Sample 98  | 0     | 623.9         | GSE12093 | Filtration   |
| Sample 99  | 0     | 717.1         | GSE12093 | Filtration   |
| ...        |       |               |          |              |
| Sample 511 | 1     | 40.5          | GSE1456  | Validation   |
| Sample 512 | 1     | 66.7          | GSE1456  | Validation   |

**TODO:** add real example to examples/ and write about it here.

## Step 2: creating configuration file

**TODO:** config format for regression.
Configuration file should contain the dictionary parameter:value in json format:

    {
        "parametor_1" : value_1,
        "parametor_2" : value_2,
        ...
        "parametor_N" : value_N
    }

For classification task, you can set values for the following parameters:
* `data_path`
    Path (absolute or relative to configuration file) to csv table of the data.

* `annotation_path`
    Path (absolute or relative to configuration file) to csv table of the data annotation.

* `n_k_path`
    Path (absolute or relative to configuration file) to a file with columns *n* and *k* defining a grid for exhaustive feature selection: *n* is a number of selected features, *k* is a length of each features subset.

* `output_dir`
    Dir for output files. If not exist, will be create.

* `feature_pre_selector`
    Name of feature pre-selection function from `./core/feature_pre_selectors.py`.

* `feature_pre_selector_kwargs`
    Dict of keyword arguments for feature pre-selector.

* `feature_selector`
    Name of feature selection function from `./core/feature_selectors.py`.

* `feature_selector_kwargs`
    Dict of keyword arguments for feature selector.

* `preprocessor`
    Name of class for data preprocessing from `sklearn.preprocessing`.

* `preprocessor_kwargs`
    Dict of keyword arguments for preprocessor initialization.

* `classifier`
    Name of class for classification from `./core/classifiers.py`.

* `classifier_kwargs`
    Dict of keyword arguments for classifier initialization.

* `classifier_CV_ranges`
    Dict defining classifier parameters which should be cross-validated. Keys are parameter names, values are iterables for grid search.

* `classifier_CV_folds`
    Number of fold for K-Folds cross-validation.

* `limit_feature_subsets`
    If true, limit the number of processed feature subsets.

* `n_feature_subsets`
    Number of processed feature subsets.

* `shuffle_feature_subsets`
    If true, processed feature subsets are selected randomly.

* `max_n`
    Maximal number of selected features.

* `max_estimated_time`
    Maximal estimated time of pipeline running.

* `scoring_functions`
    List with names for scoring functions (from `accuracy_scores.py`) which will be calculated for each classifier.

* `main_scoring_function`
    Key from scoring_functions dict defining the "main" scoring function which will be optimized during cross-validation and will be used for classifier filtering.

* `main_scoring_threshold`
    A number defining threshold for classifier filtering: classifiers with score below this threshold on training/filtration sets will not be further evaluated.

* `n_processes`
    Number of processes.

* `random_state`
    Random seed (set to an arbitrary integer for reproducibility).

* `verbose`
    If True, print running time for each pair of n, k.


## Step 3: defining a *n*, *k* grid

To estimate running time of the exhaustive pipeline and define run

```bash
python3 running_time_estimator.py /path/to/config.json max_k max_estimated_time n_feature_subsets search_max_n
```
where
* `max_k` is the maximal length of each features subset.
* `max_estimated_time` is the maximal estimated time (in hours) of single running of the exhaustive pipeline.
* `n_feature_subsets` is the number of feature subsets processed by the exhaustive pipeline (*100* is pretty good).
* `search_max_n` is *1* if you need to find the maximal number of selected features for which estimated run time of the exhaustive pipeline is less than `max_estimated_time`, and *0* otherwise.

## Step 4: running the exhaustive pipeline

When input data, configuration file and *n*, *k* grid are ready,
the exhaustive pipeline could be executed:
```bash
python3 build_classifiers.py /path/to/config.json
```
This will generate three files in the specified output folder:
* classifiers.csv: this file contains all classifiers which passed the filtration together with their quality metrics.
* summary_n_k.csv: for each pair of *n*, *k* three numbers are given: number of classifiers which passed the filtration,
number of classifiers which showed reliable performance (i.e., passed quality thresholds) on the validation set and
their ratio (in %). Low percentage of validation-reliable classifiers together with high number of 
filtration-reliable classifiers is usually associated with overfitting.
* summary_features.csv: for each feature percentage of classifiers carrying this feature 
is listed (classifiers which passed the filtration are considered).

## Step 5: generating report for a single model
To get detailed report on the specific classifier (set of features): 
* Create configuration file (use ./examples/make_classifier_summary/config.json as
   template) and set following key parameters:
    * "data_path" - path to dataset used for search of classifiers
  (relative to directory with configuration file);
    * "annotation_path" - path to annotation file (relative to directory 
      with configuration file);
    * "output_dir" - path to output directory for detailed report 
      (relative to directory with configuration file);
    * "features_subset" - set of features belonging to the classifier of interest;
* run "python3 make_classifier_summary.py <configuration file name>"    
* check the detailed report in "output_dir"

# etc
Breast and colorectal cancer microarray datasets: [OneDrive](https://eduhseru-my.sharepoint.com/:f:/g/personal/snersisyan_hse_ru/EpJztBwnLENPuLU8r0fA0awB1mBsck15t2zs7-aG4FXKNw).
