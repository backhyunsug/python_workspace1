"""
특성이 아주 많을때(고차원), 암환자 같은 30개의 특성을 갖고 
이미지 => 80 by 80 by 3 = 19200 개의 특성 
주성분분석, 원래의 특성을 가지고 원래의 특성들의 특이점을 
설명해 낼 수 있는 새로운 특성을 만들어낸다. 
19200 => 10개 뽑아내서 학습을 한다. 비슷한 결과를 가져온다. 
특성을 변환시켜서 새로운 특성을 뽑아낸다. 정확하게 뭐라 말할 수 없다.
학습시간을 줄일 수 있다. 짧은시간에 높은 학습효과를 가져온다 
비지도 학습의 일종 => 시각화 
분석을 할때, 상관관계를 확인한다. 상관계를 구한다. 
1에 가까우면 양의 상관관계 -1에 가까우면 음의상관관계 0에 가까우면 
별 관계없다. 이 상관관계를 시각화 (산포도행렬, scattermatrix)
특성대 특성의 관계를 차트로 그리는 거라서 특성이 4개면 4 * 4 = 16
특성이 30개면 30*30 = 900개 그려져야 한다. 
각 성분별로 히스토그램을 그리거나 히트맵을 활용한다. 
산포도, 히스토그램, 히트맵을 그려본다.      
"""
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris, load_breast_cancer 
import seaborn as sns 
import pandas as pd 

#iris = load_iris()
iris = load_breast_cancer()
#Bunch => DataFrame 
df1 = pd.DataFrame(iris['data'], \
                   columns=iris['feature_names'])
df1['target']=iris['target']
print(df1.head())

#seaborn 에서 산포도행렬(pairplot) -- 차트 900개 그려야 해서 30분 기다려야 함 
#sns.pairplot(df1, hue="target") 
#setosa, vesicolor, versinica 세가지 색상을 다르게 표현한다. 
#plt.show() #모든그래픽 출력은 pyplot을 사용해야 한다 

#특성이 많을 경우에 상관관계 시각화 - 히스토그램 
#악성과 양성 두개의 클래스를 갖는 데이터들의 집합  두 클래스별로 각자 데이터를 모아서 히스토그램을 그려보자 
#히스토그램이 특성의 개수만큼 나온다. 히스토그램이 데이터의 분포도를 확인 하기 좋은 차트다 
#구간을 나눠서 => 히스토그램을 그려보자 

#데이터를 악성과 양성으로 쪼개보자 
cancer = load_breast_cancer() 
cancer_data = cancer['data']
cancer_target = cancer['target']

#악성끼리 모아보자 
malignant = cancer_data[cancer_target==0]#악성종양데이터만 
benign = cancer_data[cancer_target==1]#양성종양 
print(malignant.shape)
print(benign.shape)

#히스토그램을 30개 그려보자 
#차트안에 작은 차트를 계속 만든다. 15 by 2  로 나누어서 각각의 공간에 차트하나만 
#반환값이 차트자체와 축에 대한 정보, figsize=(10, 20) 차트전체의 크기 width by height inch단위 
fig, axes = plt.subplots(15,2, figsize=(10,20))
import numpy as np 
ax = axes.ravel() #축에 대한 정보를 가져온다 
for i in range(30): #특성의 개수 30개 
    #1.구간나누기 
    count, bins = np.histogram(cancer_data[:, i], bins=50)#i번째 열에 대해서 구간을 50개로 나눠라
    #count 각구간별 데이터 개수 
    #bins 구간리스트 
    ax[i].hist(malignant[:, i], bins=bins, color='purple', alpha=0.5) #악성데이터를 보라색으로 
    ax[i].hist(benign[:, i], bins=bins, color='green', alpha=0.5) #양성데이터를 초록색으로 
    #제목 
    ax[i].set_title(cancer['feature_names'][i])
    ax[i].set_yticks(()) #y축 눈금을 없앤다  

ax[0].set_xlabel('feacture size')
ax[0].set_ylabel('fequency')
ax[0].legend(['malignant', 'benign'], loc='best')  #범수 -각각이 의미하는바
fig.tight_layout() #차트 재정렬 
plt.show()  


