import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\nabil\OneDrive\Desktop\Machine Learning\Melbourne_housing_FULL.csv")

print(df.head())
print(df.isna().sum())

useful_cols = ['Suburb', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Regionname', 'Propertycount']
df = df[useful_cols]
print(df.shape)

cols_to_fill_with_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_to_fill_with_zero] = df[cols_to_fill_with_zero].fillna(0)


df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.BuildingArea.mean())
df['YearBuilt'] = df['YearBuilt'].fillna(df.YearBuilt.mode()[0])

df.dropna(inplace = True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
print('Linear Regression Score:', round(linear.score(X_test, y_test), 2)*100, '%')

lasso = linear_model.Lasso(alpha=50, max_iter=100, tol=0.2)
lasso.fit(X_train, y_train)
print('Lasso Regression Score:', round(lasso.score(X_test, y_test), 2)*100, '%')

ridge = linear_model.Ridge(alpha=50, max_iter=100, tol=0.2)
ridge.fit(X_train, y_train)
print('Ridge Regression Score:', round(ridge.score(X_test, y_test), 2)*100, '%')
