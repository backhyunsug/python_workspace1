"""
이상치 - 통계는 확률, 특이한 값들 
        직원들 평균 연봉에 등기이사님(60억)연봉 같은 경우에는 일반적인 값의 범위를 벗어난다
        특정값 범위를 벗어나는 경우에 이를 제거하거나 평균값 또는 중간값으로 대체하려 한다 

    1) boxplot - 이상치를 시각적으로 확인한다. 
       최소값, 최대값, 4분위수(25%, 50%, 75% )
       아래 위에  동그라미 - 이상치가 있다는걸 의미 한다  
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
a = pd.Series([1,2,np.nan, 4,5,6,np.nan, 8, 9, 10])
#결측치를 100으로 대체해보기 
a [a.isna()] = 100 

#boxplot그리기 -> 보고서에 요소들의 박스플롯 
plt.figure(figsize=(8,6))
plt.boxplot(a, vert=True ) #vert-방향, True를 주면 수직으로 , False면 수평  
plt.show()

# 하한 Q1:1/4분위수 - 1.5*(Q3-Q1)  보다 작으면 이상치 
# 하한 Q3:1/4분위수 + 1.5*(Q3-Q1)  보다 크면   이상치 
# BOXPLOT의 사각형 그림이 가운데쯤 있으면 대칭형 정규분포 
# 위로 올라가면 q3쪽에 치우치는 경우에는 오른쪽으로 쏠린 그래프  
# 아래로 올라가면 Q1쪽에 치우치는 경우에는 왼쪽으로 쏠린 그래프 

def outfliers_iqr(data): #이 함수는 이상치의 하한과 상한을 반환한다 
    q1, q3 = np.percentile(data, [25, 75])  
    iqr = q3 - q1 
    lower_bound = q1 - iqr*1.5
    upper_bound = q3 + iqr*1.5
    return lower_bound, upper_bound
 
lower, upper = outfliers_iqr(a)
print("하한 ", lower)
print("상한 ", upper)

#이상치 발생했을때 삭제시킬수도 있다 
a[ a<lower ] = lower   #하한보다 낮은 값들이 있으면 대체하자 
a[ a > upper ] = upper 

plt.boxplot(a, vert=True ) #vert-방향, True를 주면 수직으로 , False면 수평  
plt.show()







