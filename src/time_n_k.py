import time

from scipy.special import binom

from src.utils import *


def predict_running_time(warm_up_n, warm_up_k, warm_up_time, grid_size, config):
    """Estimate pipeline running time for different
    values of n and k using data from a warm-up run
    
    Parameters
    ----------
    warm_up_n : int
        Value of n used for warm-up run.
    warm_up_k : int
        Value of k used for warm-up run.
    warm_up_time : float
        Number of seconds used for warm-up run.
    grid_size : int
        Upper limit on values of n for which
        function will make time predictions.
    config : dict
        Configuration dictionary.

    Returns
    -------
    pandas.DataFrame
        DataFrame with three columns: n, k and estimated time.
        The following grid is considered: 1 <= k <= n <= grid_size.
    """
    # Create dataframe with predicted running time for each n, k
    res = pd.DataFrame({
        "n": [n for n in range(1, grid_size + 1) for k in range(1, n + 1)],
        "k": [k for n in range(1, grid_size + 1) for k in range(1, n + 1)]
    })
    res["Estimated time"] = 0
    
    # Define complexity asymptotics for fitting
    # specified classifiers
    # TODO: fill this for other classifiers
    if config["model"] == "SVC":
        asymptotic_complexity = lambda n, k: k * binom(n, k)
    
    # Find multiplicative complexity constant
    # using complexity asymptotics and warm-up running time
    C = warm_up_time / asymptotic_complexity(warm_up_n, warm_up_k)
    
    # Now do extrapolation
    for i, n, k in zip(res.index, res["n"], res["k"]):
        res.loc[i, "Estimated time"] = C * asymptotic_complexity(n, k)
    
    return res


def main(config_path, warm_up_n, warm_up_k, grid_size):
    
    # Load config and input data
    config, df, ann, _ = load_config_and_input_data(config_path, load_n_k=False)
    config["verbose"] = False
    
    # Values of n and k for warm-up
    warm_up_n_k = pd.DataFrame({"n": [warm_up_n], "k": [warm_up_k]})

    # Do warm-up run and measure the running time
    start_time = time.time()
    model = initialize_classification_model(config, df, ann, warm_up_n_k)
    model.exhaustive_run()
    end_time = time.time()
    warm_up_time = end_time - start_time
    
    # Extrapolate warm-up running time to (n, k) grid
    res = predict_running_time(warm_up_n, warm_up_k, warm_up_time, grid_size, config)
    
    # Save the results
    config_dirname = os.path.dirname(config_path)
    output_dir = os.path.join(config_dirname, config["output_dir"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    res.to_csv("{}/time_estimates.csv".format(output_dir), index=None)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1] 
    
    # Values of n and k for warm-up
    # TODO: allow to pass this values from command line (default: 50 and 2)
    warm_up_n = 50
    warm_up_k = 2

    # Upper bound for n
    # TODO: allow to pass this value from command line
    grid_size = 100

    main(config_path, warm_up_n, warm_up_k, grid_size)