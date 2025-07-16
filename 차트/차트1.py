import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from matplotlib import font_manager, rc #rc가 차트가 그려질 영역이다 
#font_manager - 폰트를 폰트객체화 
#rc - 폰트를 지정할 영역 (차트영역)

#절대경로지정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/H2GTRM.TTF").get_name()
#상대경로지정 - 폰트를 복사해서 붙여놔야 한다. 
font_name = font_manager.FontProperties(fname="./fonts/H2GTRM.TTF").get_name()
print(font_name)
plt.rcParams['font.family']=font_name     #이것만 설정한다면 된다고 했는데
plt.rcParams['axes.unicode_minus'] = False #눈금에서 한글깨짐 문제 발생 

y1 = [1,2,3,4,5] #list타입 
x1 = [1,2,3,4,5] #list
x2 = x1*2        #numpy 벡터연산을 원했는데 list타입에 *2 를 하면 데이터개수 2번반복
x2 = [n*2 for n in x1] 
print(x2)
#pyplot에는 데이터를 numpy도 되고 list도 된다. 
plt.plot(np.array(x1), np.array(y1))
plt.plot(x2, y1) #x축의 데이터 개수 반드시 y축 데이터 개수는 일치  
plt.title('제목')
plt.xlabel('x축')
plt.ylabel('y축')
plt.show() 






