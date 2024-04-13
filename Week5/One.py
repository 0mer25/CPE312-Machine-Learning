import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("linear_regression_dataset.csv",sep=';')
data.head()
data.describe()
data.shape

plt.scatter(data['deneyim'],data['maas'])
plt.xlabel('Deneyim')
plt.ylabel('Maaş')
plt.show()

x=data.deneyim.values
x.shape
x

x=data.deneyim.values.reshape(-1,1)
x
x.shape

y=data.maas.values.reshape(-1,1)
y

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x,y)

b0=model.predict([[0]])
b0

b0=model.intercept_
b0

b1=model.coef_
b1

new_salary=1663+1138*11
new_salary

predict=model.predict([[11]])
predict

y_head=model.predict(x)
plt.plot(x,y_head,color='red')
plt.scatter(data['deneyim'],data['maas'])
plt.show()

from sklearn.metrics import r2_score
r2_score(y,y_head)