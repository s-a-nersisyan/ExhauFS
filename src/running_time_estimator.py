# External imports

# Internal imports
from src.utils import *


def main(config_path, max_k, max_estimated_time, n_feature_subsets, is_regressor):
    config, df, ann, n_k = load_config_and_input_data(config_path, load_n_k=False)
    config["limit_feature_subsets"] = True
    config["shuffle_feature_subsets"] = True
    config["n_feature_subsets"] = n_feature_subsets

    if is_regressor:
        model = initialize_regression_model(config, df, ann, n_k)
    else:
        model = initialize_classification_model(config, df, ann, n_k)

    df, ann = model.df, model.ann

    def get_running_time(n, k):
        _, time = model.exhaustive_run_n_k(n, k)
        return model.estimate_run_n_k_time(n, k, time)

    res = pd.DataFrame(columns=["n", "k", "Estimated time"])
    for k in range(1, max_k + 1):
        # Search max n for which estimated run time of the pipeline for classifiers
        # construction is less than max estimated time.
        start = k
        end = df.shape[1]

        time = get_running_time(start, k)
        if time >= max_estimated_time:
            n = start
        else:
            time = get_running_time(end, k)
            if time <= max_estimated_time:
                n = end
            else:
                while start < end:  # binary search
                    n = (start + end) // 2
                    time = get_running_time(n, k)
                    if n == start or n == end:
                        break
                    if time <= max_estimated_time:
                        start = n
                    else:
                        end = n

        res.loc[len(res)] = [n, k, time]

    res["n"] = res["n"].astype(int)
    res["k"] = res["k"].astype(int)

    # Save results
    output_dir = config["output_dir"]

    res.to_csv("{}/estimated_times.csv".format(output_dir), index=False)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    max_k = int(sys.argv[2])
    max_estimated_time = float(sys.argv[3])  # In hours
    n_feature_subsets = int(sys.argv[4])  # 100 is pretty good
    is_regressor = int(sys.argv[5]) == 1 if len(sys.argv) > 5 else False  # 0 or 1

    main(config_path, max_k, max_estimated_time, n_feature_subsets, is_regressor)
