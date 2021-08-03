from .utils import *


def main(config_path):
    # Load config and input data
    config, df, ann, n_k = load_config_and_input_data(config_path)

    output_dir = config["output_dir"]

    # Build regressors
    model = initialize_regression_model(config, df, ann, n_k)

    res = model.exhaustive_run()

    summary_features = get_summary_features(res)
    summary_features.to_csv("{}/summary_features.csv".format(output_dir))

    return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
