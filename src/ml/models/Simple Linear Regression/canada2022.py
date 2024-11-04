import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\canada_per_capita_income.csv")

plt.plot(df.year, df['per capita income (US$)'])
#plt.scatter(df.year, df['per capita income (US$)'])

model = linear_model.LinearRegression()
model.fit(df[['year']], df[['per capita income (US$)']])

print('Coefficient: ', model.coef_[0][0])
print('Intercept: ', model.intercept_[0])

print('prediction for 2022 is:', model.predict([[2022]])[0][0])