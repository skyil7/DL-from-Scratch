import numpy as np

x = np.array([1.0,2.0,3.0])
print(x)
print(type(x))

print()
y = np.array([2.0,4.0,6.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)

print()
A = np.array([[1,2],[3,4]])
B = np.array([10, 20])
print(A*B)

print()
X = np.array([[51,55], [14,19], [0,4]])
print(X)
X=X.flatten()#X를 1차원 배열로 평탄화
print(X)
print(X[X>15])#X>15만 출력