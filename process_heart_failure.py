import os

import numpy as np
import pandas as pd
from random import randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.accuracy_scores import min_TPR_TNR


def create_model(name, scaler, classifier):
    return name, make_pipeline(scaler, classifier)

def convert_to_our_format(df, fname):
    # rename label column
    df = df.rename(columns={"DEATH_EVENT": "Class"})

    # add random split for Validation, Training, Filtration
    df["Dataset"] = np.random.randint(0, 3, df.shape[0])
    df["Dataset type"] = df["Dataset"]

    dic = {0: "D0", 1: "D1", 2: "D2"}
    df["Dataset"] = df["Dataset"].map(dic)

    dic = {0: "Validation", 1: "Training", 2: "Filtration"}
    df["Dataset type"] = df["Dataset type"].map(dic)

    # create annotation dataframe
    anno_df = df[["Class", "Dataset", "Dataset type"]]
    anno_df.to_csv("{}.anno.csv".format(fname))

    # create features dataframe
    data_df = df[df.columns.difference(["Class", "Dataset", "Dataset type", "time"])]
    data_df.to_csv("{}.data.csv".format(fname))

def main():
    n_repeats = 100

    # load data
    fname = "data/heart_failure/heart_failure_clinical_records_dataset.csv"
    df = pd.read_csv(fname)

    # convert_to_our_format(df, fname)
    #return

    #df = df.replace('?', np.nan)
    print("Dataset shape - " + str(df.shape))
    feature_names = list(df.columns.values)
    feature_names.remove('DEATH_EVENT')
    y = df['DEATH_EVENT']

    datas = {
        "all features": df[df.columns.difference(["DEATH_EVENT", "time"])],
        "top 2 features from orig paper": df[["ejection_fraction", "serum_creatinine"]],
        "top 3 features from orig paper": df[["ejection_fraction", "serum_creatinine", "age"]],
        "top 3 features from our ranking": df[["ejection_fraction","serum_creatinine","high_blood_pressure"]]
    }

    scorer = make_scorer(matthews_corrcoef)
    #scorer = make_scorer(min_TPR_TNR)
    #scorer = make_scorer(accuracy_score)
    #scorer = make_scorer(roc_auc_score)

    scaler = StandardScaler()

    # # create list of models
    models = []
    models.append( create_model('KNN', scaler, KNeighborsClassifier()) )
    models.append( create_model('SVC', scaler, SVC()))
    models.append( create_model('LR', scaler, LogisticRegression()))
    models.append( create_model('DT', scaler, DecisionTreeClassifier()))
    models.append( create_model('GNB', scaler, GaussianNB()))
    models.append( create_model('RF', scaler, RandomForestClassifier()))
    models.append( create_model('GB', scaler, GradientBoostingClassifier()))

    # get CV performance
    i = 0
    for key in datas:
        i = i+1
        print("Case ", i, ": ", key)
        X = datas[key]
        names = []
        scores = []
        for name, model in models:
            rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats)
            score = cross_val_score(model, X, y, cv=rskfold, scoring=scorer).mean()
            names.append(name)
            scores.append(score)
        skf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
        print(skf_cross_val)
        print("")

if __name__ == "__main__":
    main()
