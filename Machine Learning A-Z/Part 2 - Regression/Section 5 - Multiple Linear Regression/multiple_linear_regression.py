# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore', dtype=np.int)
# reshaping a single column ist notwendig
print(X)
x=X[:,3].reshape(-1,1)
enc.fit(x)
encodes=enc.transform(x).toarray()
enc_T=np.transpose(encodes)
print(np.transpose(enc_T))
X=np.delete(X, -1, 1)
for i in enc_T:
    X=np.append(X, i.reshape(-1,1), axis=1)


print(X)



#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
'''
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
'''
