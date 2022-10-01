import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\carprices.csv")

dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df, dummies], axis = 1)
final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis = 1)
y = final['Sell Price($)']
X  = final.drop('Sell Price($)', axis = 1)

model = LinearRegression()
model.fit(X, y)

print(model.score(X, y))
