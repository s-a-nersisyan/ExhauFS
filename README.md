# ExhaustiveFS
Exhaustive feature selection for classication and survival analysis.

## Introduction
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

## Step 1: data preparation

Before running the tool, you should prepare two files containing actual data and its annotation. Both for classification and survival analysis csv data table should contain numerical values associated with samples (rows) and features (columns):

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

**TODO:** add real example to examples/ and write about it here.

## etc
Breast and colorectal cancer microarray datasets: [OneDrive](https://eduhseru-my.sharepoint.com/:f:/g/personal/snersisyan_hse_ru/EpJztBwnLENPuLU8r0fA0awB1mBsck15t2zs7-aG4FXKNw).
