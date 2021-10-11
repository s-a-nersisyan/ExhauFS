from sksurv.compare import compare_survival as logrank_test

from src.core.regression.accuracy_scores import dynamic_auc
from src.core.regression.accuracy_scores.dynamic_auc import dynamic_fpr_tpr
from src.core.regression.utils import structure_y_to_sksurv, plot_kaplan_mayer
from src.core.utils import float_to_latex

from src.utils.common import *


def plot_kaplan_meier_curves(
        model,
        regressor,
        df,
        ann,
        config,
        datasets,
):
    saving_format = config.get('saving_format') or 'pdf'

    min_p_value = 1
    low_value = 0
    high_value = 1
    best_quantile = 0

    for dataset_id in datasets:
        dataset, dataset_type = dataset_id.split(';')

        samples = ann[(ann['Dataset'] == dataset) & (ann['Dataset type'] == dataset_type)].index
        X = df.loc[
            samples,
            config['features_subset']
        ]
        y = ann.loc[
            samples,
            model.y_features
        ]

        if model.preprocessor:
            X = model.preprocess(X)

        risk_scores = X.to_numpy().dot(regressor.coefs.to_numpy())

        p_values = {}

        if 'Training' in dataset_type:
            for quantile in np.linspace(0, 0.25, 25):
                low_group = risk_scores <= np.quantile(risk_scores, 0.5 - quantile)
                high_group = risk_scores >= np.quantile(risk_scores, 0.5 + quantile)

                y_low = y[low_group]
                y_high = y[high_group]

                y_chunk = pd.concat([y_low, y_high], axis=0)

                group_indicators = [False] * len(y_low) + [True] * len(y_high)
                grouped_y_chunk = y_chunk.copy()
                grouped_y_chunk['group'] = group_indicators
                p_value = logrank_test(structure_y_to_sksurv(y_chunk), group_indicators)[1]

                p_values[0.5 - quantile] = p_value

                if p_value < min_p_value:
                    best_quantile = quantile
                    min_p_value = p_value

            best_quantile = 0
            low_value = np.quantile(risk_scores, 0.5 - best_quantile)
            high_value = np.quantile(risk_scores, 0.5 + best_quantile)

        low_group = risk_scores <= low_value
        high_group = risk_scores >= high_value

        y_low = y[low_group]
        y_high = y[high_group]

        y = pd.concat([y_low, y_high], axis=0)

        group_indicators = [False] * len(y_low) + [True] * len(y_high)
        grouped_y = y.copy()
        grouped_y['group'] = group_indicators

        plt.figure(figsize=(6, 6))
        plt.title('{} ({} set), $p = {}$'.format(
            dataset,
            dataset_type.lower(),
            float_to_latex(logrank_test(structure_y_to_sksurv(y), group_indicators)[1]),
        ))

        y = grouped_y[grouped_y['group'] == False]
        plot_kaplan_mayer(y, label='Low risk')

        y = grouped_y[grouped_y['group'] == True]
        plot_kaplan_mayer(y, label='High risk')

        plt.xlabel(config.get('KM_x_label') or 'Time to event')
        plt.ylabel(config.get('KM_y_label') or 'Probability of event')

        plt.ylim([0, 1.01])
        plt.yticks(np.arange(0, 1, 0.1))

        plt.tight_layout()
        plot_fname = os.path.join(
            config['output_dir'],
            "KM_{}_{}.{}".format(dataset, dataset_type, saving_format)
        ).replace('\\', '/')
        save_plt_fig(plot_fname, saving_format)
        plt.close()


def plot_roc_auc_curves(
        model,
        regressor,
        df,
        ann,
        config,
        datasets,
        year=3,
):
    saving_format = config.get('saving_format') or 'pdf'

    train_risk_scores = regressor.predict(df.loc[
        ann[
            (ann['Dataset type'] == 'Training')
            & ~((ann['Event'] == 0) & (ann['Time to event'] <= year))
        ].index,
        config['features_subset']
    ])
    train_median_risk = np.median(train_risk_scores)

    for dataset_id in datasets:
        dataset, dataset_type = dataset_id.split(';')

        samples = ann[
            (ann['Dataset'] == dataset)
            & (ann['Dataset type'] == dataset_type)
            & ~((ann['Event'] == 0) & (ann['Time to event'] <= year))
        ].index
        X = df.loc[
            samples,
            config['features_subset']
        ]
        if model.preprocessor:
            X = model.preprocess(X)

        risk_scores = regressor.predict(X)

        y_test = ann.loc[
            samples,
            model.y_features,
        ]

        fpr, tpr = dynamic_fpr_tpr(
            ann.loc[ann['Dataset type'] == 'Training', model.y_features],
            y_test,
            risk_scores,
            year=3,
        )

        auc_index = np.argmin(np.abs(risk_scores.sort_values() - train_median_risk))
        tpr_def = tpr[auc_index]
        fpr_def = fpr[auc_index]

        auc = dynamic_auc(
            ann.loc[ann['Dataset type'] == 'Training', model.y_features],
            y_test,
            risk_scores,
        )

        plt.figure(figsize=(6, 6))
        plt.title("{} ({} set), AUC = {:.2f}".format(dataset, dataset_type.lower(), auc))

        # plt.plot(fpr, tpr)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--", c="grey")

        # Plot actual FPR and TPR of regressor as a dot on ROC curve
        plt.plot(fpr_def, tpr_def, "o", color="red")

        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")

        plt.tight_layout()
        plot_fname = os.path.join(
            config['output_dir'],
            "ROC_{}_{}.{}".format(dataset, dataset_type, saving_format)
        ).replace("\\", "/")
        save_plt_fig(plot_fname, saving_format)

        plt.close()
