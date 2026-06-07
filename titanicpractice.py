import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

titanic_data = pd.read_csv(r"Machine Learning\new_titanic.csv")

#NUMBER OF PASSENGERS

print(len(titanic_data))

#COLUMN NAMES

columns = titanic_data.columns

print(columns)

#DATATYPE OF EACH COLUMN

print(titanic_data.dtypes)

#MISSING VALUES IN EACH COLUMN

print(titanic_data.isna().sum())

#AVG AGE OF PASSENGERS

age = titanic_data["Age"]

print(age.mean())

#YOUNGEST AND OLDEST PASSENGERS

highest_age = titanic_data["Age"].max()
lowest_age = titanic_data["Age"].min()

print(highest_age)
print(lowest_age)

#HOW MANY MALE AND FEMALE

print(titanic_data["Sex"].value_counts())

#HOW MANY SURVIVED

survived = titanic_data[titanic_data["Survived"] == 1]

print(survived.shape[0])






#Show all passengers younger than 18.
age = titanic_data["Age"]

print(age[titanic_data["Age"] < 18].count())

#Find passengers traveling in first class.

print(titanic_data["Pclass"].value_counts())

#Find female passengers who survived.

female_survived = titanic_data.groupby("Sex")["Survived"].value_counts()

print(female_survived)

#Find male passengers who did not survive.

male_survived = titanic_data.groupby("Sex")["Survived"].value_counts()

print(male_survived)

#Show passengers who paid fare greater than 100.

fare = titanic_data["Fare"]

high_fare = fare[titanic_data["Fare"] > 100].count()

print(high_fare)






#What is the survival rate by gender?
# What is the survival rate by passenger class?
# What is the average fare by class?
# Which embarkation port had the highest survival rate?
# What is the average age of survivors vs non-survivors?
# How many passengers were in each class?