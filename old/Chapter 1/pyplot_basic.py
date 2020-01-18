import numpy as np
import matplotlib.pyplot as plt

#데이터 준비
x = np.arange(0,6.3,0.1)#0에서 6.3까지 0.1 간격의 넘파이 배열 생성
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1,label="sin")
plt.plot(x,y2,label="cos",linestyle="--")#점선으로 그리기
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')#제목
plt.legend()    #범례 표시
plt.show()