import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Ecoli.csv")
X = df.iloc[:, 0:107]
y = df.iloc[:, 106]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)


scalar = MinMaxScaler()
X_t = X_train.iloc[:, 0:103]
X_scalar = scalar.fit(X_t.iloc[:, 0:103])
X_t1 = scalar.transform(X_t)
X_t1 = pd.DataFrame(X_t1)

X_t1["Nom (Col 104)"] = list(X_train.iloc[:, 103])
X_t1["Nom (Col 105)"] = list(X_train.iloc[:, 104])
X_t1["Nom (Col 106)"] = list(X_train.iloc[:, 105])
X_t1["Target (Col 107)"] = list(X_train.iloc[:, 106])
X_train = X_t1

target = df.shape[1]


def missing_vals(X):
    target = X.shape[1]
    for i in range(2):
        X.loc[X.iloc[:, target - 1] == i] = X.loc[X.iloc[:, target - 1] == i].fillna(
            X.loc[X.iloc[:, target - 1] == i].median())
    return X


X_train = missing_vals(X_train)
X_train = X_train.drop(X_train.columns[[target - 1]], axis=1)


def handle_outliers(X):
    X_copy = X
    X_copy = X_copy.drop(X_copy.iloc[:, [103, 104, 105]], axis=1)
    for i in X_copy:
        q1 = X_copy[i].quantile(0.25)
        q2 = X_copy[i].quantile(0.75)
        iqr = q2 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q2 + 1.5 * iqr
        for j in X_copy[i]:
            if j > upper_bound or j < lower_bound:
                X_copy[i] = X_copy[i].replace(j, np.mean(X_copy[i]))

    X.iloc[:, 0:103] = X_copy
    return X


X_train = handle_outliers(X_train)

X_train.columns = X_train.columns.astype(str)

Decision_cls = DecisionTreeClassifier(random_state=0)
Decision_cls.fit(X_train, y_train)

X_t = X_test.iloc[:, 0:103]
X_tscalar = scalar.fit(X_t.iloc[:, 0:103])
X_t1 = scalar.transform(X_t)
X_t1 = pd.DataFrame(X_t1)

X_t1["Nom (Col 104)"] = list(X_test.iloc[:, 103])
X_t1["Nom (Col 105)"] = list(X_test.iloc[:, 104])
X_t1["Nom (Col 106)"] = list(X_test.iloc[:, 105])
X_t1["Target (Col 107)"] = list(X_test.iloc[:, 106])

X_test = X_t1
target = df.shape[1]

X_test = missing_vals(X_test)
X_test = X_test.drop(X_test.columns[[target - 1]], axis=1)
X_test = handle_outliers(X_test)
X_test.columns = X_test.columns.astype(str)

d_preds = Decision_cls.predict(X_test)
#Accuracy and F1 for Decision_tree
Decision_accuracy = np.mean(cross_val_score(Decision_cls, X_test, y_test, cv=20, scoring="accuracy"))
Decision_f1 = np.mean(cross_val_score(Decision_cls, X_test, y_test, cv=20, scoring="f1"))

NB = GaussianNB()
NB.fit(X_train, y_train)
NB_pred = NB.predict(X_test)

#Accuracy and F1 for Naive Bayes
Naive_accuracy = np.mean(cross_val_score(NB, X_test, y_test, cv=20, scoring="accuracy"))
Naive_F1 = np.mean(cross_val_score(NB, X_test, y_test, cv=20, scoring="f1"))

cw = {0: 1.,
      1: 80.}
rf = RandomForestClassifier(class_weight=cw)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#Accuracy and F1 for Random Forest
rf_accuracy = np.mean(cross_val_score(rf, X_test, y_test, cv=10, scoring="accuracy"))
rf_f1 = np.mean(cross_val_score(rf, X_test, y_test, cv=10, scoring="f1"))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

#Accuracy and F1 for K nearest neighbor
knn_accuracy = np.mean(cross_val_score(knn, X_test, y_test, cv=20, scoring="accuracy"))
knn_f1 = np.mean(cross_val_score(knn, X_test, y_test, cv=20, scoring="f1"))

final_test = pd.read_csv("Ecoli_test.csv")

f_t = final_test.iloc[:, 0:103]
f_tscalar = scalar.fit(f_t.iloc[:, 0:103])
f_t1 = scalar.transform(f_t)
f_t1 = pd.DataFrame(f_t1)

f_t1["Nom (Col 104)"] = list(final_test.iloc[:, 103])
f_t1["Nom (Col 105)"] = list(final_test.iloc[:, 104])
f_t1["Nom (Col 106)"] = list(final_test.iloc[:, 105])

final_test = f_t1


final_test = handle_outliers(final_test)
final_test.columns = final_test.columns.astype(str)

knn_final = knn.predict(final_test)
knn_final = pd.DataFrame(knn_final).astype(int)

Accuracy_F1 = pd.DataFrame([[round(knn_accuracy, 3), round(knn_f1, 3)]])

knn_final["new"] = pd.Series(dtype="int")
knn_final.to_csv("s4736164.csv", header=False, index=False)
Accuracy_F1.to_csv("s4736164.csv", mode="a", header=False, index=False)
