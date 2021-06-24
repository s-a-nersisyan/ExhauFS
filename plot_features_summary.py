import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import *

# Plots feature participation curves (dependence of
# participation coefficient on the classifier arity)
# based on summary_features.csv file
#
# Input:
#   "summary_features.csv" file (generated by get_features_summary.py)
#   min_participation_ratio - minimum average rate of feature participation in k-arity classifiers
#
# Output:
#   summary_features.csv.pdf - participation curves
def main(fname, min_participation_ratio):

    # load classifiers
    df = pd.read_csv(fname.replace("\\", "/"))

    k_values = sorted(set(df["k"].unique()))
    feature_values = set(df["feature"].unique())

    feature_curves = {}
    for feature in feature_values:
        f_perc = []
        for k in k_values:
            sdf = df.loc[ (df["k"] == k) & (df["feature"] == feature) ]
            if sdf.empty:
                f_perc.append(0)
            else:
                f_perc.append( sdf.iloc[0]["percentage_classifiers"] )
        feature_curves.update({feature : f_perc})

    # plot feature curves
    num_plots = 0
    integral_thr = min_participation_ratio * len(k_values) * 100.0
    for feature in feature_curves:
        y = feature_curves[feature]
        y_integral = sum( y )
        if y_integral >= integral_thr:
            num_plots = num_plots + 1
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    # Plot several different functions...
    x = k_values
    integral_thr = min_participation_ratio * len(k_values) * 100.0
    labels = []
    for feature in feature_curves:
        y = feature_curves[feature]
        y_integral = sum( y )
        if y_integral >= integral_thr:
            plt.plot(x, y)
            labels.append(feature)

    #plt.legend(labels)
    plt.legend(labels, ncol=3, loc='lower right',
               fancybox=True, shadow=True)

    plot_fname = fname + str(".pdf")
    plt.savefig(plot_fname.replace("\\", "/"))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: /path/to/summary_features.csv min_participation_ratio", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1], float(sys.argv[2]) )
