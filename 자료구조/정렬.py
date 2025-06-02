def bin_array_sort(arr: list[int]):
    left=0
    right = len(arr)-1 
    while left<right:
        # left쪽에서 1의 위치를 찾는다 
        while arr[left] == 0 and left<len(arr)-1:
            left = left+1    
        # right 쪽에서 0의 위치를 찾는다 
        while arr[right] == 1  and right>0:
            right = right -1     
        if left<right:
            arr[left], arr[right]=arr[right],arr[left]    


for arr in ([1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1]):  
    bin_array_sort(arr)  
    print(arr)