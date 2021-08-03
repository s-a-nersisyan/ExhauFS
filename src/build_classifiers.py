from .utils import *


def main(config_path):
    # Load config and input data
    config, df, ann, n_k = load_config_and_input_data(config_path)

    # Build classifiers
    model = initialize_classification_model(config, df, ann, n_k)
    res = model.exhaustive_run()

    output_dir = config["output_dir"]

    summary_features = get_summary_features(res)
    summary_features.to_csv("{}/summary_features.csv".format(output_dir))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
