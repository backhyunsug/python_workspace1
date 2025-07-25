def searchGreater(arr):
    for i in range(0, len(arr)):
        greater = -1 #-1이라고 두고 자기보다 큰 수를 찾자 
        for j in range(i+1, len(arr)):
            if arr[j] > arr[i]:
                greater = arr[j]
                break 
        print( arr[i], "--->", greater )
        

from collections import deque
def searchGreater2(arr):
    stack = deque()
    result =[-1] * len(arr) 
    for i in range(len(arr)-1, -1, -1):
        while len(stack)!=0:
            if stack[-1]>arr[i]:
                result[i] = stack[-1] 
                break  
            else:
                stack.pop()
        
        stack.append(arr[i]) 
    
    for i in range(0, len(arr)):
        print(arr[i], " ===>", result[i] )

searchGreater2([4,5,2,25])
searchGreater([13,7,6,12])

s = deque()
s.append(2)
s.append(3)
s.append(4)
s.append(5)
print( s[-1])

