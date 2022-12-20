from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

digits = load_digits()

X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)
print('Logistic model score:', round(logistic.score(X_test, y_test)*100, 2), '%')

pca = PCA(0.95)
X_pca = pca.fit_transform(X)

# print(pca.explained_variance_ratio_)
# print(pca.n_components_)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

logistic_pca = LogisticRegression(max_iter = 1000)
logistic_pca.fit(X_train_pca, y_train)
print('Logistic model with PCA score:', round(logistic_pca.score(X_test_pca, y_test)*100, 2), '%')

pca_5 = PCA(n_components=5)
X_pca_5 = pca_5.fit_transform(X)

# print(pca.explained_variance_ratio_)

X_train_pca_5, X_test_pca_5, y_train, y_test = train_test_split(X_pca_5, y, test_size=0.2, random_state=0)

logistic_pca_5 = LogisticRegression(max_iter = 1000)
logistic_pca_5.fit(X_train_pca_5, y_train)
print('Logistic model with 5 principal components score:', round(logistic_pca_5.score(X_test_pca_5, y_test)*100, 2), '%')

