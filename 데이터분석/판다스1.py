import pandas as pd 

a = [1,2,3,4,5] 
s = pd.Series(a) #list -> Series타입으로 만들기 
print(type(s))
print(s)

b = {"a":1, "b":2, "c":3, "d":4, "e":5} 
s2 = pd.Series(b) #dict -> Series 타입으로 

print( s[0])
print( s[1])
print( s[2])
print( s[:3])
print( s[s>=3]) #조건식 가능함 

import numpy as np
#3보다크거나 같고 4보다 작거나같다
print(s[np.logical_and(s>=3, s<=4)])
print(s[[0,3,2]])

print( s2[:3])
print(s2["a"])
print(s2["a":"c"])
print(s2[["d","a","c"]])
print(s2[[4,2,1]])





