def solution(a):
    mydic = dict()
    for i in range(0, len(a)):
        if a[i] in mydic.keys():
            mydic[a[i]]+=1 
        else:
            mydic[a[i]]=1

    max = -1  
    for key in mydic.keys():
        if max < mydic[key]:
            max = mydic[key]

    cnt=0
    maxkey=0 
    for key in mydic.keys():
        if max == mydic[key]:
            maxkey = key 
            cnt+=1 

    if cnt !=1:
        return -1 
    else: 
        return maxkey
    


2
3
4
5
6
7
def solution2(array):
    while len(array) != 0:
        for i, a in enumerate(set(array)):
            print("i=", i, " a= ", a )
            array.remove(a)
        if i == 0: return a
    return -1

print( solution2( [1,2,3,3,3,4,4,4,4,4,4,4]) ) 




