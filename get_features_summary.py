import sys
import os

import pandas as pd

from utils import *

def main(output_dir):

    # load classifiers
    cls_fname = output_dir + str("/") + "classifiers.csv"
    cls_df = pd.read_csv(cls_fname.replace("\\", "/"))

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

        aaa =  111

    summary_features = pd.DataFrame(data, columns=["k", "classifiers_cnt_for_k", "gene", "percentage_classifiers"])
    summary_features.to_csv("{}/summary_features.csv".format(output_dir))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify dir with \"classifiers.csv\" file", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
