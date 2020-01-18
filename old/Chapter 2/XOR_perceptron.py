#XOR 게이트는 단층 퍼셉트론으로 구현할 수 없다. 이를 해결하기 위해 다층 퍼셉트론을 이용한다.
import numpy as np

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1.0,1.0])
    b = -0.9
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.6, 0.6])
    b = -1
    tmp = b + np.sum(x*w)
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5, -0.5])
    b = 1
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))