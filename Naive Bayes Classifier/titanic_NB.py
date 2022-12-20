import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\titanic.csv")

sex_le = LabelEncoder()
df['Sex_n'] = sex_le.fit_transform(df['Sex'])
df.Age.fillna(df.Age.mean(), inplace = True)

y = df['Survived']
X = df[['Pclass', 'Sex_n', 'Age', 'Fare']]

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = GaussianNB()
model.fit(X_train,y_train)

print('Model Score:', round(model.score(X_test, y_test), 2)*100, '%')