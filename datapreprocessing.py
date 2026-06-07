import pandas as pd
import numpy as np

data = pd.read_csv(r"Machine Learning\data.csv")

#SEPARATING DATASET

X = data.iloc[:, 0:3].values

y = data.iloc[:, 3].values

print(X, y)

#Missing Data


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


print(X)

#Encoding Data

from sklearn.preprocessing import LabelEncoder

labelen = LabelEncoder()

labelen.fit(y)

y = labelen.transform(y)

print(y)

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ctransform = ColumnTransformer(transformers= [("encoder", OneHotEncoder(), [0])], remainder= "passthrough")

X = np.array(ctransform.fit_transform(X))

print(X)

#Splitting the Dataset and Testing

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size= 0.5, random_state= 3)

print(Xtrain, Xtest, ytrain, ytest)