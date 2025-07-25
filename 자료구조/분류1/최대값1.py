def getMax1Max2(arr:list)->tuple:
    """
    함수에 대한 설명 써주기 
    """
    #max1 , max2 = arr[:2]
    max1 = arr[0]  #첫번째 데이터가 젤 크다
    max2 = arr[1]  #두번째 데이터가 두번째 크다 
    #max1이 max2보다 작은 경우를 생각해보자 
    if max1 < max2: 
        max1, max2 = max2, max1 
    print("max1", max1)
    print("max2", max2)

    for i in range(2, len(arr)):
        if max1 < arr[i]:
            max2 = max1 
            max1 = arr[i]
        elif max2 < arr[i]:
            max2 = arr[i] 
        #print("max1", max1)
        #print("max2", max2)

    return max1, max2     
arr = [3,4,2,9,8,7,6,11,12,5]
max1, max2 = getMax1Max2(arr)
print(max1, max2)