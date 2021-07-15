import pickle

import numpy as np

import matplotlib.pyplot as plt
from sksurv.compare import compare_survival as logrank_test

from src.core.regression.utils import structure_y_to_sksurv, plot_kaplan_mayer
from src.core.utils import float_to_latex
from src.utils import *


def main(config_path):
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

    model = initialize_regression_model(config, df, ann, None)
    regressor, best_params = model.fit_model(config['features_subset'])
    scores, _ = model.evaluate_model(regressor, config['features_subset'])

    # 1. Short summary on regressors accuracy on datasets
    output_dir = config['output_dir']

    # Generate report and K-M pdfs
    report = []
    report.append('Accuracy metrics:')
    report.append('//--------------------------------------------')
    for i, dataset_id in enumerate(scores):
        dataset, dataset_type = dataset_id.split(';')
        report.append('{} ({} set)'.format(dataset, dataset))
        for metr, val in scores[dataset_id].items():
            report.append('\t{:12s}: {:.4f}'.format(metr, val))

        # Plot Cox K-M curve
        X = df.loc[ann['Dataset'] == dataset, config['features_subset']]
        y = ann.loc[ann['Dataset'] == dataset, model.y_features]

        if model.preprocessor:
            X = model.preprocess(X)

        risk_scores = X.to_numpy().dot(regressor.coefs.to_numpy())
        group_indicators = risk_scores >= np.median(risk_scores)
        grouped_y = y.copy()
        grouped_y['group'] = group_indicators

        plt.figure(figsize=(6, 6))
        plt.title('{} ({} set), $p = {}$'.format(
            dataset,
            dataset,
            float_to_latex(logrank_test(structure_y_to_sksurv(y), group_indicators)[1]),
        ))

        y = grouped_y[grouped_y['group'] == True]
        plot_kaplan_mayer(y, label='High risk')

        y = grouped_y[grouped_y['group'] == False]
        plot_kaplan_mayer(y, label='Low risk')

        plt.xlabel('Time to event')
        plt.ylabel(config.get('y_label') or 'Probability of event')

        plot_fname = os.path.join(
            output_dir,
            'KM_{}.pdf'.format(dataset)
        ).replace('\\', '/')
        plt.savefig(plot_fname)
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
