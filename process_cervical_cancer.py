import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.accuracy_scores import min_TPR_TNR


def create_model(name, scaler, imputer, classifier):
    return name, make_pipeline(scaler, imputer, classifier)


def main():
    # load data
    fname = "data/cervical_cancer/risk_factors_cervical_cancer.csv"

    df = pd.read_csv(fname)
    df = df.replace('?', np.nan)
    print("Dataset shape - " + str(df.shape))
    feature_names = list(df.columns.values)
    feature_names.remove('Biopsy')

    X, y = df.loc[:, df.columns != 'Biopsy'], df['Biopsy']
    print("# of instances with y=0 - %d" % (len(y) - sum(y)))
    print("# of instances with y=1 - %d" % sum(y))

    scorer = make_scorer(min_TPR_TNR)
    #scorer = make_scorer(accuracy_score)
    #imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    #imputer = KNNImputer(n_neighbors=3, weights="distance")
    scaler = StandardScaler()
    n_repeats = 5

    # # create list of models
    models = []
    models.append( create_model('KNN', scaler, imputer, KNeighborsClassifier()) )
    models.append( create_model('SVC', scaler, imputer, SVC()))
    models.append( create_model('LR', scaler, imputer, LogisticRegression(max_iter=10000)))
    models.append( create_model('DT', scaler, imputer, DecisionTreeClassifier()))
    models.append( create_model('GNB', scaler, imputer, GaussianNB()))
    models.append( create_model('RF', scaler, imputer, RandomForestClassifier()))
    models.append( create_model('GB', scaler, imputer, GradientBoostingClassifier()))

    # get CV performance
    names = []
    scores = []
    for name, model in models:
        rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats)
        score = cross_val_score(model, X, y, cv=rskfold, scoring=scorer).mean()
        names.append(name)
        scores.append(score)
    skf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
    print(skf_cross_val)
    return


if __name__ == "__main__":
    main()
