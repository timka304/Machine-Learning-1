import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



data1 = pd.read_csv(r"Machine Learning\new_titanic.csv")


print(data1.isna().sum())




X=data1[["Pclass", "Sex", "Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"]]
y=data1["Survived"]



from sklearn.preprocessing import LabelEncoder

labelen = LabelEncoder()
labelen.fit(X["Sex"])
X["Sex"] = labelen.transform(X["Sex"])

print(X["Sex"])


from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size= 0.5, random_state= 3)

print(Xtrain, Xtest, ytrain, ytest)

