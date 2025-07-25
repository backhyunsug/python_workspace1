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
        flag = False #if문안에 들어갔었는지 확인하기 
        for j in range(0, ln-i-1):
            if arr[j] > arr[j+1]:
                flag=True #if문 안에 들어갔었다는 얘기는 데이터 변동이 있었음
                arr[j], arr[j+1] = arr[j+1], arr[j]
        if not flag:
            break #for문 종료  
        print( arr )

    print(arr)

print("bubble2")
arr = [3, 5, 7, 6, 8, 9, 25]
bubbleSort2(arr)
"""
알고리즘 -  속도가 빠르면 메모리를 많이 차지함 
          속도가 느리면 메모리 덜차지 
          trade-off 
          최근에는 메모리가 엄청 싸다 속도위주의 알고리즘을 선택한다

퀵정렬 - 재귀호출 

               8, 9, 2,4, 24, 21,  3, 6, 7, 11, 12, 13 

기준점  0~11   0번방을 기준으로 
              left =0 
              right=11 
              a[0]>a[left] 일때까지 left증가   
              left = 1
              a[0]<a[right] 일때까지 right 감소
              right = 8
              a[left] <==> a[right]

            8, 7, 2,4, 24, 21,  3, 6, 9, 11, 12, 13
            left<=right인동안 반복
            left = 4
            right = 7
            8, 7, 2,4, 6, 21,  3, 24, 9, 11, 12, 13
            left = 5 
            right = 6

            3, 7, 2,4, 6, 8,  21, 24, 9, 11, 12, 13
            left = 6
            right = 5 
            0~4               6~11
            3,7,2,4,6          21,24,9,11,12,13 

"""

arr = [5,1,6,4,8,3,7,9,2,10]
#arr[0~9] 
#arr[0~4], 기준점, a[6, 9]  
def quicksort(arr, start, end):
    #재귀호출이라 끝나는 시점 
    if start>=end:
        return 
    #기준점 
    pivot=arr[start] 
    left = start+1 
    right = end 
    print(f"left:{left} right:{right}")
    while left<=right: #left>right면 배분이 종료한거라서 
        #left증가시키면서 arr[left]가 pivot 보다 큰값을 만날때까지 
        #left가 end보다 작은동안 
        while left<=end and arr[left]<pivot:
            left+=1 
        while right>start and arr[right]>pivot:
            right-=1         
        print(f"left:{left} {arr[left]} right:{right} {arr[right]}")
        
        if left < right: #왼쪽 오른쪽이 서로 자리 바꾸어야 하는것이 있다
            arr[left], arr[right] = arr[right], arr[left]
        else: #같을 경우
            break   
    arr[start], arr[right] = arr[right], arr[start]
    print(arr)  
    quicksort(arr, start, right-1)
    quicksort(arr, right+1, end)
      

print("---------quick 정렬 --------")
quicksort(arr, 0, len(arr)-1)
print(arr)
