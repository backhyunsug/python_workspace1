import numpy as np 

#수학적으로 행렬(matrix)
m1 = np.array( [[1,2], [3,4]]) 
m2 = np.array( [[5,6], [7,8]])

"""
1 2     5 6   =   1+5   2+6      6  8              
3 4     7 8       3+7   4+8      10 12

"""
m3 = m1 + m2    # 
print(m3)

print(m1[0,0])
print(m1[0,1])
print(m1[1,0])
print(m1[1,1])

#요소의 크기 
print( m1.shape ) #tuple타입 
row, col = m1.shape 
print( row, col )
print(m1.dtype) #데이터 타입

for i in range(0,row):
    for j in range(0, col):
        print(m1[i,j],  end=' ')
    print() 



