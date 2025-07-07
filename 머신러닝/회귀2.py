import mglearn #사이킷런 책 쓴사람이 차트그리기 편하고 가끔 가짜 데이터 만들어보라고 만든 라이브러리 
X, y = mglearn.datasets.make_wave(n_samples=200)
print(X[:10])
print(y[:10])


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()   #하이퍼파라미터 없음 과대던 과소던 할수있는건 데이터셋 늘려주기 밖에 없다. 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("훈련셋 : ", model.score(X_train, y_train))
print("테스트셋 : ", model.score(X_test, y_test))
print("기울기 : ", model.coef_) 
print("절편 : ", model.intercept_)
y_pred2 = X_test * model.coef_ + model.intercept_
print(y_test)
print(y_pred)
print(y_pred2)
#다중회귀분석의 경우 가중치가 많다. 각 독립변수마다 별도의 가중치를 가져온다 
# w1x1 + w2x2 + w3x3 ......... + wnxn
"""
# (w1,w2,w3,...wn) X (x1,
                      x2, 
                      x3, 
                      x4, ......
                      xn)

"""
#Knn 이웃 회귀알고리즘 
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("훈련셋 : ", model.score(X_train, y_train))
print("테스트셋 : ", model.score(X_test, y_test))
print(y_test)
print(y_pred)




