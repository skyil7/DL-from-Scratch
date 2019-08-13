import numpy as np

A = np.array([1,2,3,4])
print(A)
print(np.ndim(A))#배열의 차원수 확인
print(A.shape)#각 차원의 값 개수 as tuple
print(A.shape[0])

B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])