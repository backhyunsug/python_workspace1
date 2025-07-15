import optuna 
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris, load_breast_cancer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC #하이퍼파라미터 개수가 많아서 선택함 
from sklearn.metrics import classification_report, accuracy_score 
# classification_report : 분류중에서도 이진분류 평가 라이브러리
# accuracy_score : 단순히 정확도 판단기준 
# GridSearchCV : 파라미터를 주면 각 파라미터 별로 전체 조합을 만들어서 다 돌려본다.
#                 
iris = load_breast_cancer() 
#iris.data, iris.target => 데이터프레임으로 
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  #0이 악성, 1이 양성 => 둘의 값을 반전하자. 나중에 모델 평가 해석시에 그게 더 편함 
y_change = np.where(y==0, 1, 0) #0->1 로 1->0으로 바꿈 

#특성의 개수 
# print( np.unique(y_change, return_counts=True) ) #return_counts=True 를 주면 value_counts 역할 
# #(array([0, 1]), array([357, 212], dtype=int64))
# print( *np.unique(y_change, return_counts=True) ) # * - unpack을 한다 
# #[0 1] [357 212]
# print( zip(*np.unique(y_change, return_counts=True) )) # * - unpack을 한다 
# #(0, 352) (1, 212) 
# print( dict(zip(*np.unique(y_change, return_counts=True) ))) 

print("유방암데이터셋 정보 : ")
print(f"특성개수 {X.shape[1]}")
print(f"샘플개수 {X.shape[0]}")
print(f"클래스분포 (0:양성 1:악성) {dict(zip(*np.unique(y_change, return_counts=True) ))} ")

#악성인 사람과 양성인 사람간의 데이터 불균형, iris 는 균형데이터 33:33:33 
#불균형데이터셋일 경우에 훈련셋과 테스트셋을 쪼갤때 그 균형을 유지하면서 쪼개라
#stratify=y_change
X_train, X_test, y_train, y_test = train_test_split(X, y_change, random_state=1234, 
                                                test_size=0.2, stratify=y_change)
print("훈련데이터셋")
print(X_train.shape, X_test.shape )
print(f"y_train분포 :  {dict(zip(*np.unique(y_train, return_counts=True) ))} ")
print(f"y_test 분포 :  {dict(zip(*np.unique(y_test, return_counts=True) ))} ")

#라이브러리 경향이 : 콜백함수를 만들어서 파라미터로 전달해주면 된다. 
#옵투나가 호출할 콜백함수를 만들어야 한다 
from sklearn.ensemble import RandomForestClassifier
def objective(trial): #변수명은 마음대로 
    #옵투나를 통해 탐색할 하이퍼파라미터 범위를 정의 한다 
    
    max_depth = trial.suggest_int('max_depth', 5, 20) #그리드서치 5,6,7,8,9,...20 => 시작,엔딩
    #트리의 최대깊이
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    #리프노드가 되시 위한 최소 샘플수 
    min_samples_split = trial.suggest_int('min_samples_split', 1, 10)
    n_estimators =  trial.suggest_int('n_estimators', 1, 10)

    model = RandomForestClassifier(max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, 
                min_samples_split=min_samples_split,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1)  #내부 프로세스 cpu개수 *2 라서 -1을 주면 알아서 최대치를 사용한다 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) #예측 정확도 
    return accuracy #반드시 마지막에 리턴해야 한다. 목적값 
    
    