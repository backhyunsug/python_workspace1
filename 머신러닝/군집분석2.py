"""
군집 - 클러스터링 
     - 결과를 안준다. 그냥 데이터만 가지고 분류를 하는데, 대강은 군집의 개수를 알려줘야 한다 
     np.random.normal(평균, 표준편차, 형태) - 평균과 표준편차를 만족하는 가우스분포를 따르는 데이터를 
     만들어낸다.    np.random.normal(173, 10, 100) 

import numpy as np 
a = np.random.normal(173, 15, 3000)
print(a[:10])

import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 6))
plt.hist(a, bins=100)
plt.show()
"""
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

X = load_iris()['data']
print(X.shape)  #y없음, y를 모른다 
print(X[:10])

#n_clusters = 3 : 몇개가 묶였는지는 알려줘야 군집분석을 효율적으로 한다
#원래 군집개수보다 적게 주면 그나마 차이가 덜한 군집끼리 합친다.  정보손실, 해석도 어렵고 
#원래 군집개수보다 많게 주면 강제로 군집을 더 쪼갠다.  
#n_clusters 를 모를때는 엘보우, 실루엣, 전문가적 견해로 

kmeans = KMeans(n_clusters = 5, random_state=42)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)
print(y_kmeans[:20])
#군집분석시 각 군집의 중심값을 가져온다 
center = kmeans.cluster_centers_ 
print("중심값", center)

#시각화 
plt.figure(figsize=(8,6))
plt.scatter(X[:, 2], X[:,3], c=y_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.scatter(center[:,2], center[:,3], c='red', s=200, marker='X', label='centroids')
plt.legend()
plt.grid()
plt.show()

#






