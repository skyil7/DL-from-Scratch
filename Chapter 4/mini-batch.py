import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def cross_entrophy_error(y, t):
    #y가 1차원이라면, reshpe로 함수의 데이터 형상을 바꿔 줌
    if y.ndim==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    #원 핫 인코딩이 아닐 시
    #return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))/batch_size

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)    #(60000, 784)
print(t_train.shape)    #(60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   #0~train_size 사이의 값을 batch_size개 추출
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

