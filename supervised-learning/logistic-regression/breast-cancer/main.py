import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

## get data and transfrom to pandas
cancer_data = load_breast_cancer()
df = sklearn_to_df(cancer_data)

# print first 5 rows and info
print(df.head())
print(df.info())

## find correlated features -> Filter Method
fig, ax = plt.subplots(figsize=(10,10)) # sample figsize in inches   
corr_grid = df.corr()
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))

# full heatmap
sns.heatmap(corr_grid, xticklabels = corr_grid.columns, yticklabels = corr_grid.columns, vmin = -1, center =0, vmax = 1, annot=False, cmap='Blues', mask=mask)
plt.show()
plt.clf()

# heatmap of features correlated to target only
sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), xticklabels=True, yticklabels = True, vmin=-1, center=0, vmax=1, annot=True, cmap='Blues')
plt.show()
plt.clf()


## determine variables for modeling (limit to 3)
corr_target = abs(corr_grid["target"])
relevant_features = corr_target[corr_target>0.5]
print(relevant_features)
# prime candidates = worst concave points, worst area, worst perimeter

## verify if all three candidates are highly correlated
print(df[["worst concave points","worst area"]].corr())
# drop worst area since it is highly correlated with worst concave points

print(df[["worst perimeter","worst concave points"]].corr())
# drop worst permiter since it is highly correlated with worst concave points

# search for other features correlated to target but not correlated to worst concave points
corr_wcp = abs(corr_grid["worst concave points"])
features_not_corr_wcp = corr_wcp[corr_wcp<0.5]
print(features_not_corr_wcp)
# none of the other features highly correlated with target were found not strongly correlated to worst concave points. Model will continue being developed with only one feature

## define X and y
X = df[['worst concave points']]
y = df.target

## stardardize X 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

## split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

## create and fit the linear regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# print intercept and coefficients
print(f"Coefficient: {lr.coef_}\nInterceopt: {lr.intercept_}")

# print out predicted outcomes, probabilities and true outcome
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:,1]
print(f'Predicted outcome: {y_pred}\nPredicted probability: {y_pred_proba}\nTrue outcome: {y_test}')

## evaluate model's performance
# Print out the confusion matrix here
print(confusion_matrix(y_test, y_pred))
# Print model accuracy, recall, precision, and f1 score
print(f'Accuracy: {accuracy_score(y_test, y_pred)}\nPrecision: {precision_score(y_test, y_pred)}\nRecall: {recall_score(y_test, y_pred)}\nF1: {f1_score(y_test, y_pred)}')
