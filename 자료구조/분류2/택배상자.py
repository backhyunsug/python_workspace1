def solution(n, w, num):
    length = n//w+1  #행의 개수 
    #2차원 배열   : 배열 구조가 아님 [ [], [], []], list of list 구조임 
    boxList = []
    for i in range(0, length):
        boxList.append( [None]*w ) 
    #print(boxList)    
    answer = 0 
    
    # 1   2  3  4 5 6  홀수행 
    # 12 11  10 9 8 7  짝수행  
    
    k=1 #계속 증가하면서 값을 저장하기 위해 1~n까지 추적한다 
    for i in range(0, length):
        if i%2==0:
            for j in range(0, w):
                if k<=n:
                    boxList[i][j]=k
                    k+=1 
        else:
            m=k+w-1 #시작위치를 키워놓고  
            for j in range(0, w):
                if m<=n:
                    boxList[i][j]=m
                    m-=1
                    k+=1 
    
    # for i in range(0, length):
    #     print( boxList[i])

    #해당데이터를 2차원 배열에서 찾아서 행과 열을 가져온다. 
    for i in range(0, length):
        if num in boxList[i]:
            row = i
            column = boxList[i].index(num)
            break
 
    #print(boxList) 
    #print( row, column)
    answer = length-row  #전체 행에서 내 위치값 뺀거 
    
    cnt=0
    for i in range(length-1, row-1, -1):
        #print(boxList[i])
        if boxList[i][column]==None:
            cnt+=1 
    #print(cnt)
    answer = answer-cnt
    return answer
    

print( solution(22,6,8) )
print( solution(13,3,3) )
