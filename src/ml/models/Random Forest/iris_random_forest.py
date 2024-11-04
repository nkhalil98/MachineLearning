import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print('Model score:', round(model.score(X_test, y_test)*100, 2), '%')

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)

plt.figure()
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')  