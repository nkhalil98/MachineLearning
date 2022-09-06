import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\canada_per_capita_income.csv")

model = linear_model.LinearRegression()
model.fit(df[['year']], df[['per capita income (US$)']])

plt.plot(df.year, df['per capita income (US$)'])

print('prediction for 2022 is:', model.predict([[2022]]))
