#1.select정렬 2
#2.버블정렬, 개선된 
#3.퀵정렬 -   

"""
select 정렬1 
9 7 25 6 8 4 3       
3 9 25 7 8 6 4     i=0  j =1,2,, ... n
3 4 25 9 8 7 6     i=1  j =2,, ... n
    6 25 9 8 7     i=2  j =3,, ... n
       7 25 9 8    i=3  j =4,, ... n
         8 25 9    i=4  j=5...
           9  25   i=5  j=6       

"""

def selectSort1(arr):
    for i in range(0, len(arr)-1):
        for j in range(i+1, len(arr)):
            if arr[i] >arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        print(arr)

arr = [9, 7, 25, 6, 8, 4, 3]
selectSort1(arr)
"""
arr = [9, 7, 25, 6, 8, 4, 3]
min  = 3
pos  = 9
9번방과 0번방을 바꿔치기 [3, 7, 25, 6, 8, 4, 9]
min = 4     
pos = 5
5번방과 2번방을 바꾼다  [3, 4, 25, 6, 8, 7, 9]
min=6                 
pos =3               [3, 4, 6, 25, 8, 7, 9]
"""

def selectSort2(arr):
    for i in range(0, len(arr)-1):
        min = arr[i] 
        pos = i 
        for j in range(i+1, len(arr)):
            if min > arr[j]:
                min = arr[j]
                pos = j 
        arr[pos], arr[i] = arr[i], arr[pos]  
        print(arr)

print("-------- selectsort2 ----------")
arr = [9, 7, 25, 6, 8, 4, 3]
selectSort2(arr)

"""
오름차순의 경우 
셀렉트정렬 - 젤 작은 사람 첫번째 반에 
           두번째 작은 사람 두번째 방에 
           세번째 작은 사람 세번째 방에
           a[i] a[j]
버블정렬 -거품  
           바로옆에사람  비교함 계속 바꿔치기 
           젤 큰사람이 뒤로 밀려
           거품이 보글거리는 느낌을 받았음 
            a[j] a[j+1]

         9, 7, 25, 6, 8, 4, 3
0 ->     9, 7 
         7, 9 
              ,25, 6
                6, 25  
                   25, 8
                    8, 25 
                       25, 4 
                       4, 25
                          25, 3
                          3   25 
i=0     7 9 6 8 4 3 25   n-1      j=0~n-i           
i=1     7 6 8 4 3 9 25   n-2      j=0~ n-i
i=2     6 7 4 3 8 9 25   n-3      j=0~ n-i
i=3     6 4 3 7 8 9 25   n-4      j=0~ n-i
i=4     4 3 6            n-5      j=0~ n-i
i=5     3 4              n-6      j=0~ n-i

if arr[j] > arr[j+1]: 이때 arr[j] arr[j+1]이 자리바꿈 

"""

def bubbleSort1(arr):
    ln = len(arr)
    for i in range(0, ln):
        for j in range(0, ln-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
        print( arr )

    print(arr)

print("----------bubble sort---------")
arr = [9, 7, 25, 6, 8, 4, 3]
bubbleSort1(arr)

"""
 3 4 6 8 7 9 25
 데이터가 대충 정렬되어 있을때  개선도
 
 """
def bubbleSort2(arr):
    ln = len(arr)
    for i in range(0, ln):
        for j in range(0, ln-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
        print( arr )

    print(arr)