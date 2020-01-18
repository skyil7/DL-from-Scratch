import numpy as np

def numerical_diff(f, x):  # 중앙 차분을 활용한 수치 미분
    h = 1e-4 # 0.0001
    #너무 작은 값을 사용하면 부동 소수점 오류 발생
    return (f(x+h) - f(x-h)) / (2*h)

def _numerical_gradient_1d(f, x):   # n차 방정식의 편미분을 통한 그래디언트 백터 계산
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 0 배열 생성

    for i in range(x.size): # i 번째 변수만 미분하고 나머지는 냅둬서 편미분
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxp = f(x)

        x[i] = tmp_val - h
        fxm = f(x)

        grad[i] = (fxp - fxm) / (2*h)
        x[i] = tmp_val # 값 복원

    return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad