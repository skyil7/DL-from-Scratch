import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

C = np.dot(A,B)

print(C)
#행렬의 곱은 아래와 같이 수행한다.
#C[0][0] = A[0][0]*B[0][0] + A[0][1] * B[1][0]
#왼쪽 행렬의 행(가로)와 오른쪽 행렬의 열(세로)를 곱하고 그 값들을 더하는 것인데
#이러한 특성 때문에 곱하는 피연산자의 순서가 바뀌면 답도 바뀐다.

C = np.dot(B,A)

print(C)

#행렬의 곱은 형상이 다른 행렬끼리도 곱할 수 있지만, A의 열과 B의 행 수가 같아야 한다.
# 3x2 행렬과 2x3 행렬의 곱은 성립하나, 3x2 행렬과 2x2 행렬의 곱은 할 수 없다.
# 3x2 행렬과 2x4 행렬의 곱도 가능하다. 이 경우 결과 C는 3x4 행렬이 된다.

print('3x2 행렬 곱하기 2x4 행렬')
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([7,8])

print(A.shape)
print(B.shape)

C=np.dot(A,B)
print(C)
print(C.shape)