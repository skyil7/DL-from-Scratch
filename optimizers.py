import numpy as np
from gradient import *

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad  # gradient 방향으로 learning rate 만큼 이동
    return x

