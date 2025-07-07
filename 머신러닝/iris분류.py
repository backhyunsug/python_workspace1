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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
print(X_train.shape)
print(X_test.shape)





