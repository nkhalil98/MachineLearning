from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

gaussian = GaussianNB()
gaussian.fit(X_train,y_train)
gaussian.score(X_test,y_test)
print('Gaussian Naive Bayes Model Score:', round(gaussian.score(X_test, y_test), 2)*100, '%')

mn = MultinomialNB()
mn.fit(X_train,y_train)
mn.score(X_test,y_test)
print('Multinomial Naive Bayes Model Score:', round(mn.score(X_test, y_test), 2)*100, '%')