import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)

print('Model score:', round(model.score(X_test, y_test)*100, 2), '%')

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)

plt.figure()
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')  
