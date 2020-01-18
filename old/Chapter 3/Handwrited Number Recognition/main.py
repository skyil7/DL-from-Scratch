import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#초기화
x,t = get_data()
network = init_network()
accuracy_cnt = 0
batch_size = 100

#각 이미지 추론
# for i in range(len(x)):
#     y=predict(network,x[i])
#     p=np.argmax(y)
#     if p == t[i]:
#         accuracy_cnt += 1

#배치 활용
for i in range(0, len(x), batch_size):#100개씩 묶어 처리
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)#axis 는 첫번째 차원의 최댓값(확률) 인덱스를 찾아줌
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("정확도 :",str(float(accuracy_cnt)/len(x)))