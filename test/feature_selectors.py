from src.utils import *

from src.core.feature_selectors import *


def main(config_path):
    config, df, ann, n_k = load_config_and_input_data(config_path)

    a = t_test(df, ann, len(df.columns))
    b = f_test(df, ann, len(df.columns))

    print(a == b)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify configuration file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
