import numpy as np 

data = np.random.rand(5)
print(data)
#저장
np.save("datafile.npy", data) #데이터 한개만 저장가능 

#불러오기 
data2=np.load('datafile.npy')
print(data2)