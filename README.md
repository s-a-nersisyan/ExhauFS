# ExhaustiveFS
Exhaustive feature selection for classication and survival analysis.

## Introduction
The main idea underlying ExhaustiveFS is the exhaustive search of feature subsets for constructing the most powerfull classification and survival regression models. Since computational complexity of such approach grows exponentially with respect to combination length, we first narrow down features list in order to make search practically feasible. Briefly, the following pipeline is implemented:
1. **Feature pre-selection:** select fixed number of features for the next steps.
2. **Feature selection:** select *n* features for exhaustive search.
3. **Exhaustive search:** iterate through all possible *k*-element feature subsets and fit classification/regression models.

Input data can consist from different batches (datasets), and each dataset should be labeled by one of the following types:
1. **Training set:** samples from training datasets will be used for tuning classification/regression models. At least one such dataset is required; if multiple given, the union will be used.
2. **Filtration set:** all tuned models will be first evaluated on training and filtration sets. If specified thresholds for accuracy are reached, model will be evaluated on validation (test) sets. The use of filtration sets is optional.
3. **Validation (test) set:** performance of models which passed filtration thresholds are then evaluated on validation sets. At least one such dataset is required; if multiple given, model will be evaluated on all test sets independently.

## etc
Breast and colorectal cancer microarray datasets: [OneDrive](https://eduhseru-my.sharepoint.com/:f:/g/personal/snersisyan_hse_ru/EpJztBwnLENPuLU8r0fA0awB1mBsck15t2zs7-aG4FXKNw).
