import numpy as np 
data1 = np.arange(1,11)
data2 = np.random.rand(10)
print(data1)
print(data2)

#키와값 쌍으로 전달, dict타입 전달시 =>저장  
np.savez('data.npz', key1=data1, key2=data2)

file = np.load("data.npz")
print(type(file))
data3 = file['key1']
data4 = file['key2']
print(data3)
print(data4)


