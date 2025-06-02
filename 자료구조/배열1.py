arr = list() 
#미리 메모리를10개만 확보한다.
for i in range(0, 10):
    arr.append(i+1)

#3번방에 데이터 88을 넣어보기 
#컴퓨터가 쓸데없는 일을 많이 해야 한다.overhead  
for i in range(9, 3, -1):
    arr[i] = arr[i-1]
arr[3]=88
print(arr)    

