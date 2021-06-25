# ExhaustiveFS
Exhaustive feature selection for classification and survival analysis.

## Table of Contents  
[Introduction](#introduction)  
[Requirements](#requirements)  
[Running ExhaustiveFS](#running-exhaustivefs)  
[Functions and classes](#functions-and-classes)  
[etc](#etc)  

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

<details>
  <summary>List of requirements</summary>

  - scipy  
  - scikit-learn  
  - numpy  
  - pandas  
  - lifelines  
  - scikit-survival  
  - xgboost  
</details>  


You can install them via:  
`pip3 install -r requirements.txt`  

# Running ExhaustiveFS

## Step 1: data preparation

Before running the tool, you should prepare three csv tables containing actual data, its annotation and *n* \ *k* grid. Both for classification and survival analysis data table should contain numerical values associated with samples (rows) and features (columns):

<details>
  <summary>Example</summary>
  
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
</details>


Sample annotation table formats are different for classification and survival analysis. For classification it should contain binary indicator of sample class (e.g., 1 for recurrent tumor and 0 for non-recurrent), dataset (batch) label and dataset type (Training/Filtration/Validation). Note that annotation should be present for each sample listed in the data table in the same order:

<details>
  <summary>Example</summary>
  
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
</details>


For survival analysis, annotation table should contain binary event indicator and time to event:
<details>
  <summary>Example</summary>
  
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
</details>


Table with *n* and *k* grid for exhaustive feature selection:  
*n* is a number of selected features, *k* is a length of each features subset.  

If you are not sure what values for *n* *k* to use, see [Step 3: defining a *n*, *k* grid](#step-3-defining-a-n-k-grid)  

<details>
  <summary>Example</summary> 
   
  | n   | k   |  
  | --- | --- |  
  | 100 | 1   |  
  | 100 | 2   |  
  | ... | ... |  
  | 20  | 5   |  
  | 20  | 10  |  
  | 20  | 15  |  
</details>


**TODO:** add real example to examples/ and write about it here.

## Step 2: creating configuration file

Configuration file is a json file containing all customizable parameters for the model (classification and survival analysis)  

<details>
  <summary>Available parameters</summary> 

  🔴!NOTE! - All paths to files / directories should be relative to the configuration file directory  
  * `data_path`
      Path to csv table of the data.

  * `annotation_path`
      Path to csv table of the data annotation.

  * `n_k_path`
      Path to a *n*/*k* grid file.

  * `output_dir`
      Path to directory for output files. If not exist, it will be created.

  * `feature_pre_selector`  
      TODO: add link and table of possible choices below  
      Name of feature pre-selection function from `./core/feature_pre_selectors.py`.

  * `feature_pre_selector_kwargs`  
      Object/Dictionary of keyword arguments for feature pre-selector function.

  * `feature_selector`  
      TODO: add link and table of possible choices below  
      Name of feature selection function from `./core/feature_selectors.py`.

  * `feature_selector_kwargs`  
      TODO: add link and table of possible choices below  
      Object/Dictionary of keyword arguments for feature selector function.

  * `preprocessor`
      Name of class for data preprocessing from `sklearn.preprocessing`.

  * `preprocessor_kwargs`
      Object/Dictionary of keyword arguments for preprocessor class initialization.

  * `model`  
    TODO: add link and table of possible choices below  
      Name of class for classification / survival analysis from `./core/classifiers.py`.

  * `model_kwargs`
      Object/Dictionary of keyword arguments for model initialization.

  * `model_CV_ranges`
      Object/Dictionary defining model parameters which should be cross-validated. Keys are parameter names, values are lists for grid search.

  * `model_CV_folds`
      Number of folds for K-Folds cross-validation.

  * `limit_feature_subsets`
      If *true*, limit the number of processed feature subsets.

  * `n_feature_subsets`
      Number of processed feature subsets.

  * `shuffle_feature_subsets`
      If *true*, processed feature subsets are selected randomly instead of alphabetical order.

  * `max_n`
      Maximal number of selected features.

  * `max_estimated_time`
      Maximal estimated pipeline running time.

  * `scoring_functions`
      List with names for scoring functions (from `accuracy_scores.py`) which will be calculated for each classifier.

  * `main_scoring_function`
      Key from scoring_functions dict defining the "main" scoring function which will be optimized during cross-validation and will be used for classifier filtering.

  * `main_scoring_threshold`
      A number defining threshold for classifier filtering: classifiers with score below this threshold on training/filtration sets will not be further evaluated.

    * `n_processes`
      Number of processes / threads to run on.
  
  * `random_state`
      Random seed (set to an arbitrary integer for reproducibility).

  * `verbose`
      If *true*, print running time for each pair of *n*, *k*.
</details>


## Step 3: defining a *n*, *k* grid

To estimate running time of the exhaustive pipeline and define adequate *n* / *k* values you can run:  
```bash
python3 running_time_estimator.py <config_file> <max_k> <max_estimated_time> <n_feature_subsets> <search_max_n> <is_regressor>
```
where
* `config_file` is the path to json configuration file.
* `max_k` is the maximal length of each features subset.
* `max_estimated_time` is the maximal estimated time (in hours) of single running of the exhaustive pipeline.
* `n_feature_subsets` is the number of feature subsets processed by the exhaustive pipeline (*100* is usually enough).
* `search_max_n` is *1* if you need to find the maximal number of selected features for which estimated run time of the exhaustive pipeline is less than `max_estimated_time`, and *0* otherwise.
* `is_regressor` is *1* if you the estimation is for the regression.

Above script calculates maximum possible values *n* / *k* for each *k*=`1...max_n` such that pipeline running time for each pair (*n*, *k*) is less then `max_estimated_time`

## Step 4: running the exhaustive pipeline

When input data, configuration file and *n*, *k* grid are ready,
the exhaustive pipeline could be executed -  
* __Classifiers__:
```bash
python3 build_classifiers.py <config_file>
```
* __Regressions__:
```bash
python3 build_regressors.py <config_file>
```

This will generate multiple files in the specified output folder:
* models.csv: this file contains all models (classifiers or regressors) which passed the filtration together with their quality metrics.
* summary_n_k.csv: for each pair of *n*, *k* three numbers are given: number of models which passed the filtration,
number of models which showed reliable performance (i.e., passed quality thresholds) on the validation set and
their ratio (in %). Low percentage of validation-reliable models together with high number of 
filtration-reliable models is usually associated with overfitting.
* summary_features.csv: for each feature percentage of models carrying this feature 
is listed (models which passed the filtration are considered).

## Step 5: generating report for a single model
To get detailed report on the specific model (== specific set of features): 
* Create configuration file (use ./examples/make_<u>(classifier | regressor)</u>_summary/config.json as
   template) and set following key parameters:
    * `data_path` - path to dataset used for search of classifiers
  (relative to directory with configuration file);
    * "annotation_path" - path to annotation file (relative to directory 
      with configuration file);
    * `output_dir` - path to output directory for detailed report 
      (relative to directory with configuration file);
    * `features_subset` - set of features belonging to the classifier of interest;
* * For classifier run `python3 make_classifier_summary.py <config_file>`   
  * For regressor run `python3 make_regressor_summary.py <config_file>`    
* Check the detailed report in `output_dir`

# Functions ans classes
- <details>
  <summary>Feature pre-selectors</summary>
  
  - <details>
    <summary>from_file</summary> 
    
    Pre-select features from a given file
    
    __name__: from_file     
    __kwargs__:   
    ```json
    {
      "sep": " "
    }
    ```
    </details>
  </details>
  
- <details>
  <summary>Feature selectors</summary>
  
  - <details>
    <summary>t_test</summary> 
    
    Select n features with the lowest p-values according to t-test
    
    __name__: t_test    
    __kwargs__:   
    ```json
    {
      "datasets": ["Training", "Filtration"]
    }
    ```
    </details>
  - <details>
    <summary>spearman_correlation</summary> 
    
    Select n features with the highest correlation with target label
    
    __name__: spearman_correlation   
    __kwargs__:   
    ```json
    {
      "datasets": ["Training", "Filtration"]
    }
    ```
    </details>
  - <details>
    <summary>from_file</summary> 
     
    Select first n features from a given file
    
    __name__: spearman_correlation   
    __kwargs__:   
    ```json
    {
      "sep": " "
    }
    ```
    </details>
  - <details>
    <summary>median</summary> 
    
    Select n features with the highest median value  
    __name__: median  
    __kwargs__:   
    ```json
    {}
    ```
    </details>
    
  ##### Regression specific selectors:
  - <details>
    <summary>cox_concordance</summary> 
       
    Select n features with the highest concordance index on one-factor Cox regression.
    
    __name__: cox_concordance  
    __kwargs__:  
    ```json
    {
      "datasets": ["Training", "Filtration"]
    }
    ```
    </details>
  - <details>
    <summary>cox_dynamic_auc</summary> 
    
    Select n features with the highest time-dependent auc on one-factor Cox regression.
  
    __name__: cox_dynamic_auc   
    __kwargs__: 
    ```json
    {
      "year": 3, // time at which to calculate auc
      "datasets": ["Training", "Filtration"]
    }
    ```
    </details>
  - <details>
    <summary>cox_hazard_ratio</summary> 
    
    Select n features with the highest hazard ratio on one-factor Cox regression.
    
    __name__: cox_hazard_ratio   
    __kwargs__:   
    ```json
    {
      "datasets": ["Training", "Filtration"]
    }
    ```
    </details>
  - <details>
    <summary>cox_likelihood</summary> 
    
    Select n features with the highest log-likelihood on one-factor Cox regression.
    
    __name__: cox_likelihood  
    __kwargs__:  
    ```json
    {
      "datasets": ["Training", "Filtration"]
    }
    ```
    </details>
  </details>
- <details>
  <summary>Classifiers</summary>
  
  - SVC
  - KNeighborsClassifier
  - RandomForestClassifier
  - XGBClassifier
  
  #### Accuracy scores
  - TPR
  - FPR
  - TNR
  - min_TPR_TNR
  </details>
- <details>
  <summary>Regressors</summary>
  
  - CoxRegression
  
  #### Accuracy scores
  - concordance_index
  - dynamic_auc
  - hazard_ratio
  - logrank
  </details>


# etc
Breast and colorectal cancer microarray datasets: [OneDrive](https://eduhseru-my.sharepoint.com/:f:/g/personal/snersisyan_hse_ru/EpJztBwnLENPuLU8r0fA0awB1mBsck15t2zs7-aG4FXKNw).
