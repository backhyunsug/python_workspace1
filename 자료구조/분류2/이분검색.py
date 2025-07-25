#이분검색은 정렬이 되어 있어야 한다. 
"""
a = [1,2,3,4,5,6,7,8,9,10]
key = 8 

1. (0+9)//2 중간값 =  4번방  5    값이  
2. 6~9  (6+9)//2   8   두번 비교하고 찾기 

없을때
key=13 
1. (0+9)//2  - 5  
2. (6+9)//2  - 7번방 8  
3. (8+9)//2  - 8번방 9 
4. (9+9)//2  - 9번방 10 
5. (10+9)//2 - not found 

left =0 
right=9 


mid = (left+rignt)//2 
만일 값이 키값보다 크면 a[mid] > key 크면   left=mid+1 
만일 값이 키값보다 크면 a[mid] <key 크면    right=mid-1 
찾았던지 left <= right 동안만 

"""
def binear_searh(arr, key): #배열과 키값 
    #1.정렬 
    arr.sort() 
    left = 0 
    right = len(arr)-1 

    while left<=right:
        mid = (left+right)//2 
        if arr[mid] < key :
            left = mid + 1 
        elif arr[mid] > key:
            right = mid -1 
        else:
            return mid 
    return -1  

a = [1,2,3,4,5,6,7,8,9,10]
print( binear_searh(a, 4) )
print( binear_searh(a, 14) )
