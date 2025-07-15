#파이프라인-> 전체적으로 유기적 연결라인  
#회귀평가와 분류평가 : 머신러닝 모델이 갖고 있는 score함수 사용 
#회귀 score :결정계수 R제곱, 분류 :정확도를 기준, 불균형셋에 대해서는 이 기준을 적용할 수 있다.  

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline #파이프라인
from sklearn.svm import SVC
from sklearn.metrics import classification_report #분류레포트 
iris = load_iris()
X = iris.data 
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
    random_state=1234)
#파이프라인 정의하기 
pipeline = Pipeline( 
    [
        ('scaler', StandardScaler() ), #데이터스케일링 
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale') )
    ]
) 

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

#성능평가 
print(classification_report(y_test, y_pred))

#그리드서치의 estimator 에 pipeline 을 전달해서 학습을 하면 된다.