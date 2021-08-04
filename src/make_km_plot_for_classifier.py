import pickle

import numpy as np
import matplotlib.pyplot as plt

from sksurv.compare import compare_survival as logrank_test

from src.core.classification.classification import ExhaustiveClassification
from src.core.regression.regression import ExhaustiveRegression
from src.core.regression.utils import structure_y_to_sksurv, plot_kaplan_mayer
from src.core.utils import float_to_latex
from src.utils import *


def main(config_path):
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

    output_dir = config['output_dir']
    model_path = config['model_path']
    saving_format = config.get('saving_format') or 'pdf'

    # Load (regressor, preprocessor) in file
    with open(model_path, 'rb') as f:
        model, preprocessor = pickle.load(f)

    for dataset, dataset_type in ann[['Dataset', 'Dataset type']].drop_duplicates().to_numpy():
        X = df.loc[
            (ann["Dataset"] == dataset) & (ann["Dataset type"] == dataset_type),
            config['features_subset']
        ].to_numpy()
        y = ann.loc[
            (ann["Dataset"] == dataset) & (ann["Dataset type"] == dataset_type),
            ExhaustiveRegression.y_features
        ]

        if preprocessor:
            X = preprocessor.transform(X)

        groups = model.predict(X).astype(bool)

        plt.figure(figsize=(6, 6))
        plt.title('{} ({} set), $p = {}$'.format(
            dataset,
            dataset_type.lower(),
            float_to_latex(logrank_test(structure_y_to_sksurv(y), groups)[1]),
        ))

        y = ann[groups == False]
        plot_kaplan_mayer(y, label='Low risk')

        y = ann[groups == True]
        plot_kaplan_mayer(y, label='High risk')

        plt.xlabel(config.get('KM_x_label') or 'Time to event')
        plt.ylabel(config.get('KM_y_label') or 'Probability of event')

        plt.ylim([0, 1.01])
        plt.yticks(np.arange(0, 1.1, 0.1))

        plt.tight_layout()
        plot_fname = os.path.join(
            output_dir,
            "KM_{}_{}.{}".format(dataset, dataset_type, saving_format)
        ).replace('\\', '/')
        save_plt_fig(plot_fname, saving_format)
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify configuration file', file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
