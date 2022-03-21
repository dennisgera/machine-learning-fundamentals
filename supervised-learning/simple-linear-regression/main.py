import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

## read csv file with honey production data
df = pd.read_csv("supervised-learning/simple-linear-regression/honeyproduction.csv")
# print first 5 rows
print(df.head())

## use groupby to get mean of total production by year
prod_per_year = df.groupby("year").totalprod.mean().reset_index()
print(prod_per_year.head())

## create variable X for year 
X = prod_per_year['year']

## reshape variable X so that it has a 2D
X = X.values.reshape(-1, 1)

## create variable y for total production
y = prod_per_year['totalprod']

## create scatter plot
plt.scatter(X, y)
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.show()


## instantiate linear regression 
regr = linear_model.LinearRegression()
## fit regression model to X and y
regr.fit(X, y)
## print slope and intercept
print(regr.coef_[0])
print(regr.intercept_)

## create y_predicted variable by predicting X with fitted model
y_predicted = regr.predict(X)
plt.plot(X, y_predicted)
plt.xlabel('Year')
plt.ylabel('Predicted Total Production')
plt.show()


## create X_future with future years 
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

## predict future total production based on fitted model 
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict)
plt.xlabel('Year')
plt.ylabel('Predicted Future Total Production')
plt.show()
plt.close()

