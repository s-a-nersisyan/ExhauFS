import sys
import pandas as pd
import seaborn as sns
import numpy as np
from sksurv.metrics import concordance_index_censored

from matplotlib import pyplot as plt


from utils import load_config_and_input_data


def main(config_path):
    scores = fit_and_score_features(data_x_numeric.values, data_y)
    pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False)
    # config, df, ann, n_k = load_config_and_input_data(config_path)
    # model = initialize_regression_model(config, df, ann, n_k)
    #
    # features = model.feature_selector(
    #     model.df,
    #     # model.df[model.pre_selected_features],
    #     model.ann,
    #     100,
    #     **model.feature_selector_kwargs
    # )
    #
    # X_train = model.df.loc[model.ann["Dataset type"] == "Training", features]
    # y_train = model.ann.loc[model.ann["Dataset type"] == "Training", ["Event", "Time to event"]]
    #
    # df = pd.concat([X_train, y_train], axis=1)
    #
    # regr = CoxPHFitter()
    #
    # regr.fit(df, duration_col='Time to event', event_col='Event')
    # regr.print_summary()
    # print(regr.predict_survival_function(df))
    # print(regr.predict_median(df))
    # print(regr.predict_partial_hazard(df))
    # #
    # # sns.lineplot(data=df, x='ITGA5', y='Time to event', hue='Event')
    # # plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
