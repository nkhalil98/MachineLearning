import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
iris = load_iris()

# without k-fold cross-validation

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2)

lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
print('Logistic regression model score:', round(lr.score(X_test, y_test)*100, 2), '%')

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print('Decision tree model score:', round(dt.score(X_test, y_test)*100, 2), '%')

svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
print('Support vector machine model score:', round(svm.score(X_test, y_test)*100, 2), '%')

rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
print('Random forest model score:', round(rf.score(X_test, y_test)*100, 2), '%')

# with k-fold cross-validation

#kf = KFold(n_splits=10)
   
# cross_val_score internally

#skf = StratifiedKFold(n_splits=10)

# def get_score(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     return model.score(X_test, y_test)
# score = []
# for train_index, test_index in skf.split(iris.data,iris.target):
#     X_train, X_test, y_train, y_test = iris.data[train_index], iris.data[test_index], iris.target[train_index], iris.target[test_index]
#     score.append(get_score(model, X_train, X_test, y_train, y_test))

print('Logistic regression model score with k-fold:', round(np.mean(cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), iris.data, iris.target, cv = 10))*100, 2), '%')
print('Decision tree model score with k-fold:', round(np.mean(cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv = 10))*100, 2), '%')
print('Support vector machine model score with k-fold:', round(np.mean(cross_val_score(SVC(gamma='auto'), iris.data, iris.target, cv = 10))*100, 2), '%')
print('Random forest model score with k-fold:', round(np.mean(cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target, cv = 10))*100, 2), '%')