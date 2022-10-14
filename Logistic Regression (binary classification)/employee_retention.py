import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\HR_comma_sep.csv")

df.head()
#sns.countplot(data = df, x = 'salary', hue = 'left')
#sns.countplot(data = df, x = 'Department', hue = 'left')

salary_dummies = pd.get_dummies(df.salary)
salary_dummies.drop('low', axis = 1, inplace = True)

X = df[['satisfaction_level', 
        'average_montly_hours', 
        'time_spend_company', 
        'Work_accident', 
        'promotion_last_5years']]

X = pd.concat([X, salary_dummies], axis = 1)
y = df.left

model = LogisticRegression(max_iter = 200)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model.fit(X_train, y_train)
model.predict(X_test)

print('Model Score:', round(model.score(X_test, y_test), 2)*100, '%')