from .utils import *


def main(config_path):
    # Load config and input data
    config, df, ann, n_k = load_config_and_input_data(config_path)

    # Build regressors
    model = initialize_regression_model(config, df, ann, n_k)

    return model.exhaustive_run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
