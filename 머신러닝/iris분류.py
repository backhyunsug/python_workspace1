"""
conda activate myenv1
conda install numpy scipy scikit-learn matplotlib ipython pandas imageio pillow graphviz python-graphviz

1.데이터준비 (전처리는 나중에는 본인이 직접 해야 한다. ) - 80%
 데이터수집, 결측치처리,이상치처리,정규화,주성분분석이나 차원축소등....., 카테고리화 원핫인코딩등  
2.데이터셋을 두개로 나눈다. 훈련셋과 테스트셋으로 나눈다. 
( 전부 다 학습을 하면 과대적합인지 과소적합인지 미래 예측력이 있는지 알 수 가 없어서
6:4 7:3 8:2 정도로 나누어서 테스트가 가능하도록, 훈련셋에만 맞춰지면 안된다. 
일반화를 위해서 쪼개야 한다)
3.알고리즘(Knn이웃 알고리즘,분류에서 가장 심플한 알고리즘)을 선택한다.
  분류알고리즘 (로지스틱회귀분석, 서포트벡터머신, 의사결정트리, 랜덤포레스트,그라디언트부스팅...)  
  을 선택해 학습을 한다. 각 알고리즘마다 성능(학습을 좀더 잘하게)을 올릴수 있는 하이퍼파라미터가 
  있는데 이걸 찾아내는 과정이 필요하다. 
4.예측을 한다. 
5.성능평가를 한다.  
"""
from sklearn.datasets import load_iris

data = load_iris() #Bunch 라는 클래스 타입 
print(data.keys())

print("타겟이름 ", data['target_names'])
print("파일명 ", data['filename'])
print("데이터설명")
print(data["DESCR"])

#2.데이터를 나누기 
X = data['data']   #ndarray 2차원배열
y = data['target'] #ndarray 1차원배열 

print(X[:10])
print(y)

#데이터를 랜덤하게 섞어서 70%추출 , train_test_split :데이터를 랜덤하게 섞어서 나눠준다
from sklearn.model_selection import train_test_split 
#tuple로 반환 , random_state인자가 시드역할, 계속 같은 데이터 내보내고 싶으면 이 값을 고정해야 한다
#test_size=0.3  그 비율대로 나뉜다 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.3)
print(X_train.shape)
print(X_test.shape)

#데이터 전체를 확인하기 위해서, 산점행렬(특성이 4개면 각 특성대 특성으로만 그릴 수 있어서 차트 4X4=16개, 
#특성이 10개가 되면 10 X 10 = 100개 차트가 만들어진다.)

#scatter_matrix 차트가 직접 노가다로 그릴수도 있고 DataFrame이 제공해준다. 아니면 Seaboarn차트를 사용하거나 
#numpy 배열을 => DataFrame으로 바꾼다
import pandas as pd 
# iris_df = pd.DataFrame(X, columns=data['feature_names']) #numpy배열과 컬러명으로 
# import matplotlib.pyplot as plt #모든 차트는  matplotlib.pyplot  가 필요하다 

# pd.plotting.scatter_matrix( iris_df, 
#                     c=y, #각 점의 색상을 지정한다. 0,1,2 각자 다른색 지정
#                     figsize=(15,15),  #차트크기 단위는 inch임 
#                     marker='o', 
#                     hist_kwds={'bins':20}, #대각선의 히스토그램의 구간 개수 
#                     s=60, #점의 크기 
#                     alpha=0.8  #투명도 1이 불투명, 0으로 갈수록 투명하다 
# )
#plt.show()

#Knn이웃알고리즘 : 내옆집에 누가 사느냐를 확인하는것 , 거리로 가장 가까운 거리에 누가 있느냐 
#                거리를 재는 방식이 유클리드 기하학을 사용한다 
#                이웃을 몇개까지볼거냐를 선택할 수 있다. 
#                이웃의 개수를 지정할 수 있다.   대부분의 홀수개를 지정하는 경우가 많다. 
#                회귀 분류 둘다 가능 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2) #이웃의 개수 3개 
#학습을 시작한다. 한습한 내용은 모델 자체가 갖고 있고 충분히 모델의 하이퍼파라미터가 지정되어서 
#최대한의 학습효과를 얻었다고 생각하면 모델을 저장해놓고 나중에 불러와서 다시 쓸 수 있다.
model.fit(X_train, y_train) 

#예측하기 
y_pred = model.predict(X_test) #테스트셋으로 예측한 데이터를 반환한다. 
#본래의 테스트셋인 y_test와 비교해본다 
print(y_pred)
print(y_test)

#평가하기 
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))

#클래스이름으로 출력
# class_names = list(data['target_names'])
# for i, j in zip(y_pred, y_test):
#     print("예측 :{:20s} 실제:{:20s}".format(class_names[i], class_names[j]))


#로지스틱 분류 : 2진, 다중분류, R은 2진분류만 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
model.fit(X_train, y_train) 
print("로지스틱일때")
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))
print(model.coef_)
print(model.intercept_)

#의사결정트리 (회귀와 분류 모두 가능하다 )
#필연적으로 과대적합이 된다. 알고리즘 자체가 과대적합으로 간다 
#의사결정트리 알고리즘은 특성의 중요도 파악용임 
from sklearn.tree import DecisionTreeClassifier
#트리시작이 랜덤이라서 , 시드를 잡아주지 않으면 만들어질때마다 다르게 나온다 
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)
print("의사결정트리")
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))
print("특성의 중요도 ", model.feature_importances_) #특정의 중요도 

#수평막대 차트를 그려보자 : 중요도를 그려보자 
import matplotlib.pyplot as plt
import numpy as np  
def treeChart(model, feature_name):
    #수평막대개수 구하기 : 특성의 개수만큼 구하면 된다. 
    n_features = len(model.feature_importances_)
    #barh - 수형막대그래프 
    plt.barh(np.arange(n_features), model.feature_importances_ ,
             align="center")
    plt.yticks(np.arange(n_features), feature_name)#y축 단위 
    plt.ylim(-1, n_features) #눈금 범위 
    plt.show() 

treeChart(model, np.array(data['feature_names']))

#랜덤포레스트 : 의사결정트리를 랜덤하게 많이 만들어서 평균값 따져서 예측하는 
#             앙상블의 일종, 과대적합의 위험을 갖고 있다
from sklearn.ensemble import RandomForestClassifier
#random_state 꼭 지정해줘야 한다  
#max_depth : 트리의 깊이를 막자 
#n_estimators : 결정트리를 몇개까지 만들까 너무 크면 시간이 많이 걸린다. 
#너무 작으면 과대적합문제가 발생한다. 일반화 
#모델을 생성할때 전달되는 파라미터를 하이퍼파라미터라고 하고 이 값들을 적절히 활용해서 과대적합도 과소적합도 
#막아서 일반화를 해야 한다. 
model = RandomForestClassifier( random_state=0, 
                              max_depth=3, 
                              n_estimators=1000)
model.fit(X_train, y_train)
print("랜덤포레스트=====================================")
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))
