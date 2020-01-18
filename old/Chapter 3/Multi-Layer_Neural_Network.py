import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

#입력층
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print('입력계층의 가중치 합 :',A1)
print('1층의 변환 신호 :',Z1)

#1층
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print('1층의 가중치 합 :',A2)
print('2층의 변환 신호 :',Z2)

#2층 -> 출력층
W3 = np.array([[0.1, 0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2, W3) + B3
Y = A3

print('2층의 가중치 합 :',A3)
print('항등 함수를 거친 최종 출력 :',Y)