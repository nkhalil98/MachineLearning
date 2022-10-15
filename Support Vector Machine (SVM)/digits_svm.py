import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train, y_train)
print('rbf Model score:', round(rbf_model.score(X_test, y_test)*100, 2), '%')


linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)
print('linear Model score:', round(linear_model.score(X_test, y_test)*100, 2), '%')

# other parameters to tune: C (Regularization) and gamma