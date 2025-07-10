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

np.random.seed(42)
x1 = np.random.normal(0,  1,(50,2))  #형태가 tuple로  2차원으로 
x2 = np.random.normal(5,  1,(50,2))  #형태가 tuple로  2차원으로 
x3 = np.random.normal(2.5,1,(50,2))  #형태가 tuple로  2차원으로 

#데이터를 합친다. concatenate-인자들이 dataframe이어야 한다 
#np.vstack을 사용해서 세로로 결합한다 

X = np.vstack( (x1, x2, x3) ) #매개변수를 tuple로 전달해야 한다.  
print(X.shape)  #y없음, y를 모른다 
print(X[:10])

#n_cluserters = 3 : 몇개가 묶였는지는 알려줘야 군집분석을 효율적으로 한다 
kmeans = KMeans(n_clusters = 3, random_state=42)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)
print(y_kmeans[:20])
#군집분석시 각 군집의 중심값을 가져온다 
center = kmeans.cluster_centers_ 
print("중심값", center)

#시각화 
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:,1], c=y_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.scatter(center[:,0], center[:,1], c='red', s=200, marker='X', label='centroids')
plt.legend()
plt.grid()
plt.show()








