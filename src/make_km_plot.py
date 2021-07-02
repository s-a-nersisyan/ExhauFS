import pickle

import numpy as np

import matplotlib.pyplot as plt
from sksurv.compare import compare_survival as logrank_test

from src.core.regression.regression import ExhaustiveRegression
from src.core.regression.utils import structure_y_to_sksurv, plot_kaplan_mayer
from src.core.utils import float_to_latex
from src.utils import *


def main(config_path):
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

    output_dir = config['output_dir']
    model_path = config['model_path']

    # Load (regressor, preprocessor) in file
    with open(model_path, 'rb') as f:
        regressor, preprocessor = pickle.load(f)

    for i, dataset in enumerate(ann['Dataset'].drop_duplicates().to_numpy()):
        X = df.loc[ann['Dataset'] == dataset, config['features_subset']]
        y = ann.loc[ann['Dataset'] == dataset, ExhaustiveRegression.y_features]

        if preprocessor:
            X = pd.DataFrame(
                preprocessor.transform(X),
                index=X.index,
                columns=X.columns,
            )

        risk_scores = X.to_numpy().dot(regressor.coefs.to_numpy())
        group_indicators = risk_scores >= np.median(risk_scores)
        grouped_y = y.copy()
        grouped_y['group'] = group_indicators

        plt.figure(figsize=(6, 6))
        plt.title('K-M, {} ({} set), $p = {}$'.format(
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify configuration file', file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
