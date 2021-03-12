import sys
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from utils import *
from core.accuracy_scores import TPR, FPR


def save_SVM_feature_importances(classifier, fname):
    imp = classifier.coef_[0]
    names = config["features_subset"]
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align="center")
    plt.yticks(range(len(names)), names)
    plt.title("SVM classifier feature importances")
    plt.xlabel("Feature weight")
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    # Load config and input data
    config_path = sys.argv[1]
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)

    # If necessary, add ROC AUC metric
    if "ROC_AUC" not in config["scoring_functions"]:
        config["scoring_functions"].append("ROC_AUC")

    # Fit classifier
    model = initialize_classification_model(config, df, ann, None)
    classifier, best_params, preprocessor = model.fit_classifier(config["features_subset"])
    scores, _ = model.evaluate_classifier(classifier, preprocessor, config["features_subset"])

    # 1. Short summary on classifiers accuracy on datasets
    config_dirname = os.path.dirname(config_path)
    output_dir = os.path.join(config_dirname, config["output_dir"]).replace("\\","/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generate report and roc_auc pdfs
    report = []
    report.append("Accuracy metrics:")
    report.append("//--------------------------------------------")
    for i, dataset in enumerate(scores):
        dataset_type = ann.loc[ann["Dataset"] == dataset].loc[:, "Dataset type"].values[0].lower()
        report.append("{} ({} set)".format(dataset, dataset_type))
        for metr, val in scores[dataset].items():
            report.append("\t{:12s}: {:.4f}".format(metr, val))
            
            # Plot ROC curve
            if metr == "ROC_AUC":
                X = df.loc[ann["Dataset"] == dataset, config["features_subset"]].to_numpy()
                y = ann.loc[ann["Dataset"] == dataset, "Class"].to_numpy()
                
                if preprocessor:
                    X = preprocessor.transform(X)
                
                y_score = classifier.predict_proba(X)
                fpr, tpr, thresholds = roc_curve(y, y_score[:, 1])
                
                y_pred = classifier.predict(X)
                tpr_def = TPR(y, y_pred)
                fpr_def = FPR(y, y_pred)
                
                plt.figure(figsize=(6, 6))
                plt.title("ROC curve, {} ({} set)".format(dataset, dataset_type))

                # default threshold classifier performance
                plt.plot(fpr_def, tpr_def,'o', color='red')
                # TODO: maybe need to add some test nearby
                
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], "--", c="grey")
                
                plt.xlim([-0.01, 1.01])
                plt.ylim([-0.01, 1.01])

                plt.xlabel("1 - Specificity")
                plt.ylabel("Sensitivity")
                
                plot_fname = os.path.join(
                    output_dir,
                    "ROC_{}.pdf".format(dataset)
                ).replace("\\","/")
                plt.savefig(plot_fname)
                plt.close()

        report.append("")

    # Save feature importances report
    if config["classifier"] == "SVC":
        fi_fname=os.path.join(output_dir,"feature_importances.pdf").replace("\\","/")
        save_SVM_feature_importances(classifier, fi_fname)

    # Save report in file
    rep_fname=os.path.join(output_dir,"report.txt").replace("\\","/")
    with open(rep_fname, "w") as f:
        for item in report:
            f.write("%s\n" % item)
            print("%s" % item)

    # Save (classifier, preprocessor) in file
    model_fname = os.path.join(output_dir, "model.pkl").replace("\\","/")
    with open(model_fname, "wb") as f:
        pickle.dump((classifier, preprocessor), f)
