import numpy as np
import matplotlib.pyplot as plt

array2=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("Vector:",array2)
a=array2.reshape(3,5)
print("Two dimensional array:",a)


print("shape:",a.shape)
print("dimension:",a.ndim)
print("data type:",a.dtype.name)
print("size:",a.size)
print("type:",type(a))

x=np.array([1,2,3])
y=np.array([4,5,6])
print(x+y)
print(x-y)
print(x**2)

a=np.array([1,2,3])
d=a.copy()
print(d)
b=a
c=a
b[0]=5
print(a,b,c)

a=np.array([1,2,3,4,5,6,7])
print(a[0])
print(a[0:4])

reverse_array=a[::-1]
print(reverse_array)

b=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(b)
print(b[1,1])
print(b[:,1])
print(b[1,:])
print(b[1,1:4])
print(b[-1,:])
print(b[:,-1])


data = np.random.randn(1000)

plt.hist(data,bins=30)
plt.title("histogram")
plt.xlabel("Values")
plt.ylabel("Frequencies")
plt.show()

sizes=[30,25,15,10,5,5]
plt.pie(sizes)
plt.title("Pie Chart")
plt.show()