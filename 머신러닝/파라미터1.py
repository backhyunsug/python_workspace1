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
X = iris.data 
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

#파라미터 설정하기 
param_grid={
    'C':[0.1, 1, 10, 100], #오차허용범위를 조절한다. 내부적으로 오차를 조율하는데 이 값 범위내서 움직임 
                           #값이 크면 과대적합 위험 
    'gamma':[1, 0.1, 0.01, 0.001], #커널의 영향 범위, 클수록 과대적합
    'kernel':['rbf', 'linear'] #비선형(데이터가 비선형일때 좋음), 선형구조(데이터가 선형일때) 
}
#{'C': 10, 'gamma': 1, 'kernel': 'linear'}

svc = SVC(C=10, gamma=1, kernel='linear') 
# #직접학습을 하지 않고 GridSearchCV에게 맡긴다 
# grid = GridSearchCV(estimator=svc, param_grid=param_grid, 
#                     cv=5, verbose=2, scoring='accuracy')
# #estimator - 학습모델 객체전달 
# #param_grid - 파라미터로 쓸 대상
# #cv - kfold검증, 데이터가 충분히 많으면 10까지 가능 
# #verbose - 출력로그 0:없음, 1:간단 2:자세히 
# #scoring - 평가 수단을 정확도에 맞춤 
# grid.fit(X_train, y_train)
# print("최적의 파라미터")
# print(grid.best_params_) 
# print("최고스코어")
# print(grid.best_score_)
#svc = grid.best_estimator_  #학습해놓은 모델 갖고 와서  
svc.fit(X_train, y_train)
print("훈련셋   평가 : ", svc.score(X_train, y_train))
print("테스트셋 평가 : ", svc.score(X_test,  y_test))



