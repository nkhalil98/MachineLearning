import pandas as pd
from sklearn import linear_model
from text_to_num import text2num
import math

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\hiring.csv")

df.experience = df.experience.apply(lambda x: text2num(x, 'en') if type(x) is str else x)
df.experience = df.experience.fillna(0)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(math.floor(df['test_score(out of 10)'].median()))

model = linear_model.LinearRegression()
model.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])
