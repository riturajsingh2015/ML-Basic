# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # first : indicates we select all the rows and select all columns except last one.
y = dataset.iloc[:, 3].values # first : indicates we select all the rows and select purschased column which has index 3.


# Taking care of missing data
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') # take mean along axis=0 that is columns and fit missing having NaN with the mean
#imp_mean.fit(X[:,1:3])  #imputer= imputer.fit(X[:,1:3]) #tell imputer object what things to handle for filling empty data , fit imputer on all rows which have NaN but only on columns from 1 to 2
X[:,1:3]=(imp_mean.fit_transform(X[:,1:3])) # this line actually transforms  X[:,1:3] into what we require and returns that4


from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler


#-----------Cleaning and changing the data so as to make it suitabke for our model---------#

enc = OneHotEncoder(handle_unknown='ignore', dtype=np.int)
# reshaping a single column ist notwendig
x=X[:,0].reshape(-1,1)
enc.fit(x)
encodes=enc.transform(x).toarray()
enc_T=np.transpose(encodes)
X=np.delete(X, 0, 1)
for i in list(range(3))[::-1]:
    X=np.insert(X, [0], enc_T[i].reshape(-1,1), axis=1)


label_encoder = LabelEncoder()
#note here we didnot use OneHotEncoder which we should have used but instead we used LabelEncoder because there is a slight catch in here
# countries should be treated equally that is why onehotEncoder
# but purschase i.e yes no should also be encoded using OneHotEncoder
# the catch is from X we predict y i.e different rows of X give rise to y
# so ordering mattters in case of X when there exists one but its not the same for y jus y should be encoded appropriately
# so as to pass to the model

y = label_encoder.fit_transform(y)


#-----------Now model preparation starts---------#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# feature scaling is done in order to bring the features in the same scale range so that no one feature is dominant than the other
print(X_train)
sc_X= StandardScaler()
 # fit()  calculates the parameters (e.g. μ and σ in case of StandardScaler)
#and saves them as an internal objects state. Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
#You do that on the training set of data i.e calculate the parameters (e.g. μ and σ in case of StandardScaler)
#But then you have to apply the same transformation(same μ and σ so you donot fit it again just transform it.) to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast. But you have to use the same two parameters μ and σ (values) that you used for centering the training set.
#https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print(X_test)
