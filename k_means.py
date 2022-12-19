import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)

sepal = df[['sepal length (cm)', 'sepal width (cm)']]
petal = df[['petal length (cm)', 'petal width (cm)']]

km_sepal = KMeans(n_clusters=3)
km_petal = KMeans(n_clusters=3)


# data preprocessing
scaler = MinMaxScaler()

scaler.fit(sepal[['sepal length (cm)']])
sepal['sepal length (cm)'] = scaler.transform(sepal[['sepal length (cm)']])

scaler.fit(sepal[['sepal width (cm)']])
sepal['sepal width (cm)'] = scaler.transform(sepal[['sepal width (cm)']])

scaler.fit(petal[['petal length (cm)']])
petal['petal length (cm)'] = scaler.transform(petal[['petal length (cm)']])

scaler.fit(petal[['petal width (cm)']])
petal['petal width (cm)'] = scaler.transform(petal[['petal width (cm)']])

y_sepal = km_sepal.fit_predict(sepal)
y_petal = km_petal.fit_predict(petal)

sepal['cluster_num'] = y_sepal
petal['cluster_num'] = y_petal

sepal_1 = sepal[sepal.cluster_num == 0]
sepal_2 = sepal[sepal.cluster_num == 1]
sepal_3 = sepal[sepal.cluster_num == 2]

plt.scatter(sepal_1['sepal length (cm)'],sepal_1['sepal width (cm)'],color='blue')
plt.scatter(sepal_2['sepal length (cm)'],sepal_2['sepal width (cm)'],color='green')
plt.scatter(sepal_3['sepal length (cm)'],sepal_3['sepal width (cm)'],color='red')
plt.show()
plt.clf()

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(sepal)
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()
plt.clf()

petal_1 = petal[petal.cluster_num == 0]
petal_2 = petal[petal.cluster_num == 1]
petal_3 = petal[petal.cluster_num == 2]

plt.scatter(petal_1['petal length (cm)'],petal_1['petal width (cm)'],color='blue')
plt.scatter(petal_2['petal length (cm)'],petal_2['petal width (cm)'],color='green')
plt.scatter(petal_3['petal length (cm)'],petal_3['petal width (cm)'],color='red')
plt.show()

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(petal)
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

