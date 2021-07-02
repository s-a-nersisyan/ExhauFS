from .utils import *


def main(config_path):
    # Load config and input data
    config, df, ann, n_k = load_config_and_input_data(config_path)

    output_dir = config["output_dir"]

    # Build regressors
    model = initialize_regression_model(config, df, ann, n_k)
    res = model.exhaustive_run()

    # Summary table #2: for each feature calculate
    # percentage of reliable regressors which use it
    feature_counts = {}
    for features_subset in res.index:
        for feature in features_subset.split(";"):
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    summary_features = pd.DataFrame(
        {"percentage_regressors": feature_counts.values()},
        index=feature_counts.keys()
    )
    summary_features["percentage_regressors"] *= 100 / len(res) if len(res) else 1
    summary_features = summary_features.sort_values("percentage_regressors", ascending=False)
    summary_features.index.name = "gene"

    summary_features.to_csv("{}/summary_features.csv".format(output_dir))

    return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
