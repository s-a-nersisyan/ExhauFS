from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

# Alias for sklearn ROC AUC function
# This is an exception for general signature:
# instead of y_true and y_pred, this function
# accepts y_true and y_score (see sklearn
# documentation for more details).
ROC_AUC = roc_auc_score


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


def FPR(y_true, y_pred):
    """False positive rate

    Parameters
    ----------
    y_true : array-like
        List of true class labels
    y_pred : array-like
        List of predicted class labels

    Returns
    -------
    float
        False positive rate
    """
    return 1 - TNR(y_true, y_pred)


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
