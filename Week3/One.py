import numpy as np
import pandas as pd
import os

titanic_train = pd.read_csv("C:\Genel\GithubRepos\CPE312-Machine-Learning\Files/train.csv")
titanic_train.shape
titanic_train.head(5)

categorical = titanic_train.dtypes[titanic_train.dtypes == "object"].index
print(categorical)
titanic_train[categorical].describe()

titanic_train["Ticket"][0:15]

titanic_train["Ticket"].describe()
del titanic_train["Ticket"]

new_Pclass = pd.Categorical(titanic_train["Pclass"] , ordered=True)
new_Pclass = new_Pclass.rename_categories(["Class1","Class2","Class3"])     
new_Pclass.describe()

titanic_train["Pclass"] = new_Pclass
titanic_train["Cabin"].unique()

char_cabin = titanic_train["Cabin"].astype(str)
new_Cabin = np.array([cabin[0] for cabin in char_cabin])
new_Cabin = pd.Categorical(new_Cabin)
new_Cabin .describe()

titanic_train["Cabin"] = new_Cabin
dummy_vector = pd.Series([1,None,3,None,7,8])
dummy_vector.isnull()

titanic_train["Age"].describe()
missing = np.where(titanic_train["Age"].isnull() == True)
missing

len(missing[0])

titanic_train.hist(column='Age',
                   figsize=(9,6),
                   bins=20)


new_age_var = np.where(titanic_train["Age"].isnull(),
                       28,                      
                       titanic_train["Age"])

titanic_train["Age"] = new_age_var 
titanic_train["Age"].describe()

titanic_train.hist(column='Age',
                   figsize=(9,6),
                   bins=20)

