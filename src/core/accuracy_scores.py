"""Accuracy scores

Every accuracy score function takes two array-like
objects y_true and y_pred, containing true and predicted
class labels, and returns a single number: score.

Each scoring function have the following signature:
def score(y_true, y_pred):
    # Code
    return score
"""

from .regression.accuracy_scores import *
from .classification.accuracy_scores import *
