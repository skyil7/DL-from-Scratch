import sys, os
sys.path.append(os.pardir)
import numpy as np
from activations import softmax
from losses import cross_entropy
from gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규 분포로 weight 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy(y, t)

        return loss

net = simpleNet()
print('최초 가중치')
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print('최초 예측')
print(p)

np.argmax(p) # 최댓값 인덱스

t = np.array([0,0,1])
net.loss(x,t)

dW = numerical_gradient(lambda w:net.loss(x, t), net.W)
print('기울기')
print(dW)

net.W = dW

print('경사법 이후 예측')
print(net.predict(x))