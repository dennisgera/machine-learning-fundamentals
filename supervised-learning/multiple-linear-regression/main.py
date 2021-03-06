import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis = pd.read_csv('supervised-learning/multiple-linear-regression/tennis_stats.csv')
print(tennis.columns)
print(tennis.info())
print(tennis.head())

# perform exploratory analysis here:
# get heatmap of correlated features
corr_grid = tennis.corr()
sns.heatmap(corr_grid, xticklabels = corr_grid.columns, yticklabels = corr_grid.columns, vmin = -1, center =0, vmax = 1, annot=False, cmap='Blues')
plt.show()

# plot highly correlated feature to winnings outcome
plt.plot(tennis['ServiceGamesWon'], tennis['Winnings'], 'o')
plt.xlabel('ServiceGamesWon')
plt.ylabel('Winnings')
plt.show()
plt.close()


## perform single feature linear regressions here:
features = tennis[['BreakPointsOpportunities']]
outcome = tennis[['Winnings']]

## use scikit's train_test_split function
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)
model_score = model.score(features_test, outcome_test)
print(model_score)
prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.show()
plt.close();



## perform two feature linear regressions here:
features = tennis[['BreakPointsOpportunities',
'FirstServeReturnPointsWon']]
outcome = tennis[['Winnings']]


## use scikit's train_test_split function
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)
model_score = model.score(features_test, outcome_test)
print(model_score)
prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.show()
plt.close();


## perform multiple feature linear regressions here:
features = tennis[['BreakPointsOpportunities', 'FirstServeReturnPointsWon', 'BreakPointsFaced', 'ServiceGamesWon']]
outcome = tennis[['Winnings']]

## use scikit's train_test_split function
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)
model = LinearRegression()
model.fit(features_train, outcome_train)
model_score = model.score(features_test, outcome_test)
print(model_score)

prediction = model.predict(features_test)

plt.scatter(outcome_test, prediction, alpha=0.4)
plt.show()
plt.close()




















