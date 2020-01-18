import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x>0
    #[True, False, True, Flase] 식의 np.ndarray로 저장
    return y.astype(np.int)#1,0으로 바꿔서 배열로 리턴

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)#y축의 범위 지정
plt.title('step function')
plt.show()