from sklearn.datasets import fetch_openml #fetch 로 되는 함수들은 너무 커서 별도의 다운을 요구한다.
boston = fetch_openml("boston", version=1)
print(type(boston))
print(boston.keys())
print(boston["DESCR"])

X = boston["data"]
y = boston["targeet"]

print(X.shape)
print(X[:10])
print(y[:10])
