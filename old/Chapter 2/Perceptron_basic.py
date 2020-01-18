import numpy as np
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1

#세타를 이항하여 bias(편향)으로 표현할 수도 있다.
def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,1))

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])#AND와 가중치가(w와 b) 다르다.
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp<=0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1, x2])
    w = np.array([1.0, 1.0])
    b = -0.9
    tmp = np.sum(w*x) + b
    if tmp<=0:
        return 0
    else:
        return 1
