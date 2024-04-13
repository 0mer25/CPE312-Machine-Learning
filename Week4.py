import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv")
data.head()

data.info()
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
M =data[data.diagnosis=="M"]
B =data[data.diagnosis=="B"]
M.info()
B.info()

plt.scatter(M.radius_mean,M.area_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.area_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.show()

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()

data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
data.diagnosis

y=data.diagnosis.values
x_data=data.iloc[:,1:3].values
x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

from  sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_head=knn.predict(x_test)
y_head

print("when k is {}, accuracy of KNN classification {} " .format(3,knn.score(x_test,y_test)))

test_accuracy=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    test_accuracy.append(knn2.score(x_test,y_test))

plt.figure(figsize=(5,5))
plt.plot(range(1,15),test_accuracy)
plt.title("k values vs accuracy")
plt.xlabel("k labels")
plt.xlabel("accuracy")
plt.grid()
plt.show()
print("best accuracy is {} with K ={}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))