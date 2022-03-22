import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('transactions.csv')
print(df.head())
print(df.info())

# Summary statistics on amount column
print(df['amount'].describe())

# Create isPayment field
df['isPayment'] = df['type'].apply(lambda x: 1 if x == 'PAYMENT' or x == 'DEBIT' else 0)

# Create isMovement field
df['isMovement'] = df['type'].apply(lambda x: 1 if x == 'CASH_OUT' or x == 'TRANSFER' else 0)

# Create accountDiff field - Our hypothesis, in this case, being that destination accounts with a significantly different value could be suspect of fraud.
df['accountDiff'] = abs(df['oldbalanceOrg'] - df['oldbalanceDest'])

# Create features and label variables
features = df[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = df['isFraud']

# Split dataset
features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.3, random_state = 27)

# Normalize the features variables
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

# Fit the model to the training data
lr = LogisticRegression()
lr.fit(features_train, label_train)

# Score the model on the training data
score_train = lr.score(features_train, label_train)
print('Score - training data:', score_train)

# Score the model on the test data
score_test = lr.score(features_test, label_test)
print('Score - testing data:', score_test)

# Print the model coefficients
print(f'Coefficients: {lr.coef_}\nIntercept: {lr.intercept_}')

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([1334290.29, 0.0, 1.0, 221293.92])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1, transaction2, transaction3, your_transaction), axis=0)

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
new_transactions_predictions = lr.predict(sample_transactions)
print(new_transactions_predictions)

# Show probabilities on the new transactions
new_transactions_predictions_proba = lr.predict_proba(sample_transactions)[:,1]
print(new_transactions_predictions_proba)