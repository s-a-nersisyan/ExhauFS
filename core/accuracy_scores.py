"""Accuracy scores

Every accuracy score function takes two array-like
objects y_true and y_pred, containing true and predicted
class labels, and returns a single number: score.

Each scoring function have the following signature:
def score(y_true, y_pred):
    # Code
    return score
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def TPR(y_true, y_pred):
    """True positive rate (sensitivity)
    
    Parameters
    ----------
    y_true : array-like
        List of true class labels
    y_pred : array-like
        List of predicted class labels

    Returns
    -------
    float
        True positive rate
    """
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TP / (TP + FN)


def TNR(y_true, y_pred):
    """True negative rate (specificity)
    
    Parameters
    ----------
    y_true : array-like
        List of true class labels
    y_pred : array-like
        List of predicted class labels

    Returns
    -------
    float
        True negative rate
    """
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TN / (TN + FP)


def min_TPR_TNR(y_true, y_pred):
    """Minimum of true positive rate (sensitivity) and
    true negative rate (specificity).
    
    Parameters
    ----------
    y_true : array-like
        List of true class labels
    y_pred : array-like
        List of predicted class labels

    Returns
    -------
    float
        min(true positive rate, true negative rate)
    """
    return min(TNR(y_true, y_pred), TPR(y_true, y_pred))

def ROC_AUC(y_true, y_proba):
    """ Area under curve.
    
    Parameters
    ----------
    y_true : array-like
        List of true class labels
    y_proba : array-like
        List of predicted probabilities of 1-labels

    Returns
    -------
    float
        area under curve
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    return roc_auc