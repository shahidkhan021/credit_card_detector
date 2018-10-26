#credit card detection
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn


data = pd.read_csv("creditcard.csv")

# print(data.columns)
# print(data.shape)
# print(data.describe())
data.sample(frac=0.1,random_state=1)

# print(data.shape)
# data.hist(figsize = (20,20))
# plt.show()
Fraud = data[data["Class"] == 1]
Valid = data[data["Class"] == 0]

fruad_fraction = len(Fraud)/len(Valid)

# print(fruad_fraction)
# print(len(Fraud))
# print(len(Valid))

columns = data.columns.tolist()

columns = [c for c in columns]


target = "Class"

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define random state
state = 1

#define the outlier detection methods
classifier = {
    "IsolationForest": IsolationForest(max_samples=len(X),
                                       contamination=fruad_fraction,
                                    random_state= state),
    "LocalOutlierFactor":LocalOutlierFactor(n_neighbors=20,
                                            contamination=fruad_fraction)

}


n_outliers = len(Fraud)

for i,(clf_name,clf) in enumerate(classifier.items()):
    if clf_name == "LocalOutlierFactor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negetive_outlier_factor_
    else:
        y_pred = clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)


    # Reshape the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()
    print('{} : {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))



