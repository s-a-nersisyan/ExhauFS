from src.utils import *

# Computes feature summary file from list of classifiers
#
# Input:
#   fname - file with classifiers (like classifiers.csv)
#
# Output:
#   summary_features.csv
#
#       each line contains following fields
#        "k"                      - arity of classifier (number of features in classifier)
#        "classifiers_cnt_for_k"  - number of classifiers with such arity
#        "feature"                - name of feature which is present in classifier
#        "percentage_classifiers" - percentage of classifiers with this "feature" among all classifiers with arity "k"
def main(fname):

    output_dir = os.path.dirname(os.path.abspath(fname))

    # load classifiers
    cls_df = pd.read_csv(fname.replace("\\", "/"))

    k_values = set(cls_df["k"].unique())
    feature_values = set()
    for index, row in cls_df.iterrows():
        for feature in row['features'].split(";"):
            feature_values.add(feature)

    # feature summary for each k value
    data = []
    for k in k_values:
        sdf = cls_df.loc[cls_df["k"] == k]
        cl_cnt = sdf.shape[0]

        # Summary table: for each feature calculate
        # percentage of reliable classifiers which use it
        feature_counts = {}
        for index, row in sdf.iterrows():
            features_subset = row["features"]
            for feature in features_subset.split(";"):
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        feature_counts = dict( sorted(feature_counts.items(), key=lambda item: item[1], reverse=True) )

        for feature in feature_counts:
            percentage_classifiers = feature_counts[feature] * 100 / cl_cnt if cl_cnt else 1
            data.append([k, cl_cnt, feature, percentage_classifiers])

    summary_features = pd.DataFrame(data, columns=["k", "classifiers_cnt_for_k", "feature", "percentage_classifiers"])
    summary_features.to_csv("{}/summary_features.csv".format(output_dir))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify filename with classifiers (e.g. /tmp/classifiers.csv)", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
