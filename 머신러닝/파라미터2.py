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
    'svc':{
        'C':[0.1, 1, 10, 100], #오차허용범위를 조절한다. 내부적으로 오차를 조율하는데 이 값 범위내서 움직임 
                            #값이 크면 과대적합 위험 
        'gamma':[1, 0.1, 0.01, 0.001], #커널의 영향 범위, 클수록 과대적합
        'kernel':['rbf', 'linear'] #비선형(데이터가 비선형일때 좋음), 선형구조(데이터가 선형일때)
    }, 
    'random_forest':{
        'n_estimators':[50, 100, 200], #트리를 여러개만든다. 50, 100,200 개 만들어봐라 
        'max_depth':[None, 3, 10, 20], #최대깊이
        'min_samples_split':[2,5,10] #가지치기 개수 
    },
    'gradient_boosting':{
        'n_estimators':[50, 100, 200], #트리를 여러개만든다. 50, 100,200 개 만들어봐라 
        'max_depth':[None, 3, 10, 20], #최대깊이
        'learning_rate':[0.01, 0.1, 0.2] #학습률  
    }
}

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
results = []
models = {
    'svc':SVC(),
    'random_forest':RandomForestClassifier(),
    'gradient_boosting':GradientBoostingClassifier()
}

results = {}
models = {
    'svc': SVC(),
    'random_forest': RandomForestClassifier(random_state=42),
    'gradient_boosting': GradientBoostingClassifier(random_state=42)
}

#모델에 저장된요소마다 파라미터를 이용해 적용해 본다. 
for model_name, model in models.items():
    print(f"Running GridSearchCV for {model_name}...")
    grid = GridSearchCV(estimator=model, 
                        param_grid=param_grid[model_name], cv=5, verbose=2, scoring='accuracy')
    grid.fit(X_train, y_train)
    results[model_name] = {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'best_model': grid.best_estimator_
    }

# 4. 테스트 데이터에 대한 성능 평가
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print("Best Parameters:", result['best_params'])
    print("Best Cross-Validation Score:", result['best_score'])
    
    # 테스트 데이터로 예측
    y_pred = result['best_model'].predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Test Set Accuracy:", accuracy_score(y_test, y_pred))


#옵투나 => 베이지안정리, GridSearchCV업그레이드 버전 