import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


data = pd.read_csv("creditcard.csv")

data.sample(frac= 0.1,random_state=1)
Fraud = data[data["Class"] == 1]
Valid = data[data["Class"] == 0]

fraud_factor = len(Fraud)/len(Valid)

columns = data.columns.tolist()

columns = [c for c in columns]
target = "Class"
X = data[columns]
Y = data[target]

classifiers = {
    "IsolationForest" : IsolationForest(max_samples=len(X),contamination = fraud_factor,random_state=1),
    "LocalIsolators"  : LocalOutlierFactor(n_neighbors=20,contamination=fraud_factor)
}


outlier = fraud_factor

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "IsolationFores":
        y_pred = clf.fit_predict(X)
        score_pred = clf.negetive_outlier_factor_
    else:
        y_pred = clf.fit(X)
        score_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    y_pred[y_pred == 1] == 0
    y_pred[y_pred == -1] == 1

    n_errors = (y_pred != Y).sum()
    print("{} : {}".format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))