"""
지도학습 - 출력결과를 알고 있을때 
비지도학습 - 결과를 모른다. 라벨링이 없는 학습니다. 
          반드시 그렇지는 않지만 대체 뭘 분석해야 할지 모르고 결과도 예측이 안되고 
          지도학습 전단계에 데이터 분석용으로 많이 사용한다. 
          분류 -> 군집 
          분류 : 기린하고 병아리 사진을 주면 분류의 경우에는 기린일 확률 0.7 병아리일 확률이 0.3입니다.
          군집 : 대충 2 클래스가 있는 걸로 보여요  클래스1이 될 확률 0.7 클래스2가 될 확률이 0.3입니다 
                대충 몇개의 클래스가 존재한다 정도, 
          스케일링(4가지, StandardScalar, RobustScalar, MinMaxScalar, Normalize) 
          차원축소 - 고차원데이터(특성이 아주 많은경우 - 사진, 3차원 => 1차원  ) 시간도 많이 걸리고 
                    사진처럼 정밀하게 그리는 경우와 캐리커쳐(특징만가지고)를 그렸을때 캐리커쳐가 더 빠르다
                    때로는 차원축소로 할때 성과가 더 좋은경우도 있다. 
          주성분분석 - 분산을 가지고 분산이 큰거로 주성분을 찾아낸다. 암환자데이터의 경우 특성이 많을 경우 
                    유용하다 
          연관성분석(장바구니분석)-월마트의  기저귀하고 맥주가 같은 장바구니에 많이 들어갈줄 모르고 분석 
                  - 숫자로 다 바꾸고 -> 벡터로 다 바꾸고         
                  - 숫자로 바꾸기만 하면 연산을 가능하다 

강화학습 - 게임, 알파고, 채찍과 당근, 파이썬 3.9 책이 전부 절판, 

사이킷런 에서  지도학습    fit  -> predict 
            비지도학습   fit  -> transform  
"""
#스케일링을 하기위한 가짜데이터 만들기
data = {
    "feature1":[160,   165,  170,  175, 180,  155,   190, 172, 168, 178],
    "feature2":[3000, 3200, 3500, 4900, 4800, 6000, 2800, 3300, 5600, 4700],
    "feature3":[3,     2,     1,   4,   4 ,   6,      12, 13,     11,  6]
}

import pandas as pd 
df = pd.DataFrame(data)
print(df)

#StandardScalar : 데이터가 정규분포를 따른다고 가정할때 많이 쓰임, 모델특성이 스케일링에 민감할때 
#                 서포트벡터머신, 로지스틱, 딥러닝 일때 유용하다 , 이상치에 민감하다  
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
ss = StandardScaler() #객체 생성 
df_scaled = ss.fit_transform(df)  #학습하고 바로 변경된값 반환 
print(df_scaled)

#RobustScaler : 데이터에 이상치에 많으면 StandardScaler 대신 쓸 수 있다. 
rb = RobustScaler()
df_scaled = rb.fit_transform(df)  #학습하고 바로 변경된값 반환 
print(df_scaled)

#MinMaxScaler : 특성값의 범위가 명확히 0~1 사이에 와야할때(이미지,특정신경망)
rb = MinMaxScaler()
df_scaled = rb.fit_transform(df)  #학습하고 바로 변경된값 반환 
print(df_scaled)

#Normalizer - 주로 텍스트분석에 유용하다. 클러스터링(군집분석)
rb = Normalizer()
df_scaled = rb.fit_transform(df)  #학습하고 바로 변경된값 반환 
print(df_scaled)

#별짓 다해도 데이터가 많아야 한다. 