from sksurv.metrics import cumulative_dynamic_auc

from core.regression.utils import structure_y_to_sksurv


def dynamic_auc(y_train, y_test, y_pred, years=3):
    structured_y_train = structure_y_to_sksurv(y_train)
    structured_y_test = structure_y_to_sksurv(y_test)

    return cumulative_dynamic_auc(
        structured_y_train,
        structured_y_test,
        y_pred,
        [years*365],
    )[0][0]
