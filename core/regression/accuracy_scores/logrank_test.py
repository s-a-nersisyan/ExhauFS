from sksurv.compare import compare_survival as logrank_test

from core.regression.utils import structure_y_to_sksurv


def logrank(y_true, x, model_coefs):
    risk_scores = x.multiply(model_coefs)
    group_indicators = (risk_scores >= risk_scores.median()).astype(bool)
    print(risk_scores)
    print(group_indicators)

    return logrank_test(structure_y_to_sksurv(y_true), group_indicators).pvalue
