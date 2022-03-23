# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# Load the data
obesity = pd.read_csv("feature-engineering/wrapper-methods/obesity.csv")

# Inspect the data
print(obesity.head())

# Split the data into predictor variables and an outcome variable
X = obesity.iloc[:,:-1]
y = obesity.iloc[:,-1]

# Create a logistic regression model
lr = LogisticRegression(max_iter=1000)

# Fit the logistic regression model
lr.fit(X, y)

# Print the accuracy of the model
print(f'Model accuracy considering all available features: {lr.score(X, y)}')

# Create a sequential forward selection model
sfs = SFS(
  estimator=lr,
  k_features=9,
  forward=True,
  floating=False,
  scoring='accuracy',
  cv=0
)

# Fit the sequential forward selection model to X and y
sfs.fit(X, y)

# Inspect the results of sequential forward selection
print(f"SFS results: {sfs.subsets_[9]}")

# See which features sequential forward selection chose
print(f"SFS feature names: {sfs.subsets_[9]['feature_names']}")

# Print the model accuracy after doing sequential forward selection
print(f"SFS accuracy: {sfs.subsets_[9]['avg_score']}")

# Plot the model accuracy as a function of the number of features used
plot_sfs(sfs.get_metric_dict())
plt.show()
plt.clf()

# Create a sequential backward selection model
sbs = SFS(
  estimator=lr,
  k_features=7,
  forward=False,
  floating=False,
  scoring='accuracy',
  cv=0
)

# Fit the sequential backward selection model to X and y
sbs.fit(X, y)

# Inspect the results of sequential backward selection
print(sbs.subsets_[7])

# See which features sequential backward selection chose
print(f"SBS feature names: {sbs.subsets_[7]['feature_names']}")

# Print the model accuracy after doing sequential backward selection
print(f"SBS accuracy: {sbs.subsets_[7]['avg_score']}")

# Plot the model accuracy as a function of the number of features used
plot_sfs(sbs.get_metric_dict())
plt.show()
plt.clf()

# Get feature names
features = X.columns

# Standardize the data
X = pd.DataFrame(StandardScaler().fit_transform(X))

# Create a recursive feature elimination model
rfe = RFE(
  estimator=lr,
  n_features_to_select=8
  )

# Fit the recursive feature elimination model to X and y
rfe.fit(X, y)

# See which features recursive feature elimination chose
rfe_features = [f for (f, support) in zip(features, rfe.support_) if support]
print(f"RFE feature names: {rfe_features}")

# Print the model accuracy after doing recursive feature elimination
print(f"RFE accuracy: {rfe.score(X, y)}")

