import pickle

import numpy as np

import matplotlib.pyplot as plt
from sksurv.compare import compare_survival as logrank_test

from src.core.regression.utils import structure_y_to_sksurv, plot_kaplan_mayer
from src.core.utils import float_to_latex
from src.utils import *


def main(config_path):
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

    saving_format = config.get('saving_format') or 'pdf'

    model = initialize_regression_model(config, df, ann, None)
    regressor, best_params = model.fit_model(config['features_subset'])
    scores, _ = model.evaluate_model(regressor, config['features_subset'])

    regressor.print_summary()

    # 1. Short summary on regressors accuracy on datasets
    output_dir = config['output_dir']

    # Generate report and K-M pdfs
    report = []
    report.append('Accuracy metrics:')
    report.append('//--------------------------------------------')

    print(model.df, model.ann)
    # TODO: remove (test)
    min_p_value = 1
    low_value = 0
    high_value = 1
    best_quantile = 0
    for i, dataset_id in enumerate(scores):
        dataset, dataset_type = dataset_id.split(';')
        report.append('{} ({} set)'.format(dataset, dataset_type.lower()))
        for metr, val in scores[dataset_id].items():
            report.append('\t{:12s}: {:.4f}'.format(metr, val))

        # Plot Cox K-M curve
        X = df.loc[
            (ann['Dataset'] == dataset) & (ann['Dataset type'] == dataset_type),
            config['features_subset']
        ]
        y = ann.loc[
            (ann['Dataset'] == dataset) & (ann['Dataset type'] == dataset_type),
            model.y_features
        ]

        if model.preprocessor:
            X = model.preprocess(X)

        risk_scores = X.to_numpy().dot(regressor.coefs.to_numpy())

        # TODO: remove (test)
        p_values = {}

        if 'Training' in dataset:
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

        # group_indicators = risk_scores >= np.median(risk_scores)
        # grouped_y = y.copy()
        # grouped_y['group'] = group_indicators

        low_group = risk_scores <= low_value
        high_group = risk_scores >= high_value

        y_low = y[low_group]
        y_high = y[high_group]

        y = pd.concat([y_low, y_high], axis=0)

        group_indicators = [False] * len(y_low) + [True] * len(y_high)
        grouped_y = y.copy()
        grouped_y['group'] = group_indicators

        print(dataset, best_quantile, low_value, high_value, np.median(risk_scores), logrank_test(structure_y_to_sksurv(y), group_indicators)[1])

        plt.figure(figsize=(6, 6))
        plt.title('{} ({} set), p-value vs quantile'.format(
            dataset,
            dataset_type.lower(),
        ))
        plt.plot(p_values.keys(), p_values.values())
        plt.savefig(os.path.join(
            output_dir,
            "p_values_{}_{}.pdf".format(dataset, dataset_type)
        ).replace('\\', '/'))

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
            output_dir,
            "KM_{}_{}.{}".format(dataset, dataset_type, saving_format)
        ).replace('\\', '/')
        save_plt_fig(plot_fname, saving_format)
        plt.close()

    report.append('')

    # Save report in file
    rep_fname = os.path.join(output_dir, 'report.txt').replace('\\', '/')
    with open(rep_fname, 'w') as f:
        for item in report:
            f.write('%s\n' % item)
            print('%s' % item)

    # Save (regressor, preprocessor) in file
    model_fname = os.path.join(output_dir, 'model.pkl').replace('\\', '/')
    with open(model_fname, 'wb') as f:
        pickle.dump((regressor, model.preprocessor), f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify configuration file', file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
