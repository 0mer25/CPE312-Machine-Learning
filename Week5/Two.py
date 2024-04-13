import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("advertising.csv")
df.head()

df.shape
df.describe()
df.info()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['TV','Radio','Newspaper']],df['Sales'],test_size=0.2,random_state=100)
X_train

X_test.shape

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

model.intercept_
model.coef_
y_pred=model.predict(X_test)
y_pred
y_pred.shape

new_df=pd.DataFrame({'Actural_values':y_test,'Prediction_values':y_pred})
new_df.head()

from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)