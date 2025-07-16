import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from matplotlib import font_manager, rc #rc가 차트가 그려질 영역이다 
import seaborn as sns 

#font_manager - 폰트를 폰트객체화 
#rc - 폰트를 지정할 영역 (차트영역)


#절대경로지정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/H2GTRM.TTF").get_name()
#상대경로지정 - 폰트를 복사해서 붙여놔야 한다. 
font_name = font_manager.FontProperties(fname="./fonts/H2GTRM.TTF").get_name()
print(font_name)


#seaborn 의 스타일 설정을 rcParams 고치는거보다 먼저 해야 한글이 안깨진다. 
sns.set_style('whitegrid') #{darkgrid, whitegrid, dark, white, ticks}

plt.rcParams['font.family']=font_name     #이것만 설정한다면 된다고 했는데
plt.rcParams['axes.unicode_minus'] = False #눈금에서 한글깨짐 문제 발생 

x = np.linspace(0, 2, 100) #구간을 나눈다. 0~2사이를 100개 쪼개서 값을 np.array로 준다 
print(x[:20]) 

#pyplot에는 데이터를 numpy도 되고 list도 된다. 
plt.plot(x, x,    label='선형',   color='g') #초록색 
plt.plot(x, x**2, label='2차', color='b') #파랑색 
plt.plot(x, x**3, label='3차', color='r') #빨간색 

plt.title('제목')
plt.xlabel('x축')
plt.ylabel('y축')
plt.legend(loc="best")  #범주를 니가 봐서 적당한 위치에 출력해라 
plt.show() 






