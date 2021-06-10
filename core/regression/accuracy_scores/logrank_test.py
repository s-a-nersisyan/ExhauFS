import numpy as np
from sksurv.compare import compare_survival as logrank_test

from core.regression.utils import structure_y_to_sksurv


def logrank(y_true, x, model_coefs):
    risk_scores = x.to_numpy().dot(model_coefs.to_numpy())
    group_indicators = risk_scores >= np.median(risk_scores)

    return -np.log10(logrank_test(structure_y_to_sksurv(y_true), group_indicators)[1])
