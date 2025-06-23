#merge - 병합 
#이미 정렬되어 있는 배열을 합쳐서 정렬한다 
a = [5, 7, 9, 11, 12, 23, 27, 34]
b = [3, 4, 5, 7, 9, 11, 13, 19, 23, 27, 27, 33, 34, 35, 40, 42]
"""
i <- a꺼         a[i] == b[j]  c.append(a[i]) i+=1 j+=1  
j <- b꺼         a[i] < b[j]   a[i]를 내보낸다.  i+=1 
                 a[j] > b[i]   b[j]를 내보낸다. j+=1

                 마지막은 남은 배열을 모두 C로 내보내면 된다. 
"""
c = [3,4,5,7,9,11,12,13,19,23,27,33,34]
"""
디비 없을때 파일로 작업 - 갱신 

인사파일                              수정파일 
1   홍길동  2024-03-09  ......        1   D 2025-06-16
                                     101 I 장길산 2025-06-12  

0"""

def merge(a, b):
    c = [] 
    i=0
    j=0 
    while i<len(a) and j<len(b):
        if a[i] == b[j]:
            c.append(a[i])
            i+=1 
            j+=1 
        elif a[i] < b[j]:
            c.append(a[i])
            i+=1 
        else:
            c.append(b[j])
            j+=1  

    while i<len(a):
        c.append(a[i])
        i+=1 
    
    while j<len(b):
        c.append(b[j])
        j+=1 

    return c 
    
print( merge(a,b))
