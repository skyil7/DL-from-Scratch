import numpy as np

def MSE(y, l):
    return 0.5 * np.sum((y-l)**2)

def cross_entropy_weak(y, l):
    if y.ndim == 1:
        l = l.reshape(1, l.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(l * np.log(y + 1e-7)) / batch_size

def cross_entropy(y, t):# batch 호환
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size