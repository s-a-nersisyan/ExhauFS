import pickle

from src.utils.common import *
from src.utils.summary_regressors import plot_kaplan_meier_curves, plot_roc_auc_curves


def main(config_path):
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

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

    for i, dataset_id in enumerate(scores):
        dataset, dataset_type = dataset_id.split(';')
        report.append('{} ({} set)'.format(dataset, dataset_type.lower()))
        for metr, val in scores[dataset_id].items():
            report.append('\t{:12s}: {:.4f}'.format(metr, val))

    plot_kaplan_meier_curves(model, regressor, df, ann, config, scores.keys())
    plot_roc_auc_curves(model, regressor, df, ann, config, scores.keys())

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
