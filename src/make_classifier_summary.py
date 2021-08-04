import pickle

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.utils import *
from src.core.accuracy_scores import TPR, FPR


def save_model_feature_importances(config, classifier, fname):
    if config["model"] == "SVC":
        imp = classifier.coef_[0]
    if config["model"] == "RandomForestClassifier":
        imp = classifier.feature_importances_

    names = config["features_subset"]
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align="center")
    plt.yticks(range(len(names)), names)
    plt.title("Classifier feature importances")
    plt.xlabel("Feature weight")
    plt.savefig(fname)
    plt.close()


def main(config_path):

    # Load config and input data
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

    saving_format = config.get('saving_format') or 'pdf'

    # If necessary, add ROC AUC metric
    if "ROC_AUC" not in config["scoring_functions"]:
        config["scoring_functions"].append("ROC_AUC")

    # Fit classifier
    model = initialize_classification_model(config, df, ann, None)
    classifier, best_params = model.fit_model(config["features_subset"])
    scores, _ = model.evaluate_model(classifier, config["features_subset"])

    # 1. Short summary on classifiers accuracy on datasets
    output_dir = config["output_dir"]

    # Generate report and roc_auc pdfs
    report = []
    report.append("Accuracy metrics:")
    report.append("//--------------------------------------------")
    for i, dataset_id in enumerate(scores):
        dataset, dataset_type = dataset_id.split(';')
        report.append("{} ({} set)".format(dataset, dataset_type))
        for metr, val in scores[dataset_id].items():
            report.append("\t{:12s}: {:.4f}".format(metr, val))

            # Plot ROC curve
            if metr == "ROC_AUC":
                X = df.loc[
                    (ann["Dataset"] == dataset) & (ann["Dataset type"] == dataset_type),
                    config["features_subset"]
                ].to_numpy()
                y = ann.loc[
                    (ann["Dataset"] == dataset) & (ann["Dataset type"] == dataset_type),
                    "Class"
                ].to_numpy()

                if model.preprocessor:
                    X = model.preprocessor.transform(X)

                y_score = classifier.predict_proba(X)
                fpr, tpr, thresholds = roc_curve(y, y_score[:, 1])

                y_pred = classifier.predict(X)
                tpr_def = TPR(y, y_pred)
                fpr_def = FPR(y, y_pred)

                plt.figure(figsize=(6, 6))
                plt.title("{} ({} set), AUC = {:.2f}".format(dataset, dataset_type.lower(), val))

                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], "--", c="grey")

                # Plot actual FPR and TPR of classifier as a dot on ROC curve
                plt.plot(fpr_def, tpr_def, "o", color="red")

                plt.xticks(np.arange(0, 1.1, 0.1))
                plt.yticks(np.arange(0, 1.1, 0.1))

                plt.xlim([-0.01, 1.01])
                plt.ylim([-0.01, 1.01])

                plt.xlabel("1 - Specificity")
                plt.ylabel("Sensitivity")
                
                plt.tight_layout()
                plot_fname = os.path.join(
                    output_dir,
                    "ROC_{}_{}.{}".format(dataset, dataset_type, saving_format)
                ).replace("\\", "/")
                save_plt_fig(plot_fname, saving_format)

                plt.close()

        report.append("")

    # Save feature importances report
    if config["model"] in ["SVC", "RandomForestClassifier"]:
        fi_fname = os.path.join(output_dir, "feature_importances.pdf").replace("\\", "/")
        save_model_feature_importances(config, classifier, fi_fname)

    # Save report in file
    rep_fname = os.path.join(output_dir, "report.txt").replace("\\", "/")
    with open(rep_fname, "w") as f:
        for item in report:
            f.write("%s\n" % item)
            print("%s" % item)

    # Save (classifier, preprocessor) in file
    model_fname = os.path.join(output_dir, "model.pkl").replace("\\", "/")
    with open(model_fname, "wb") as f:
        pickle.dump((classifier, model.preprocessor), f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
