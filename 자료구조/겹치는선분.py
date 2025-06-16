def solution(lines):
    answer = 0
    
    temp = [[0]*200 for _ in range(3) ]
    # for line in temp:
    #     print(line)

    i=0
    for start, end in lines:
        for j in range(start+100, end+100):
            temp[i][j]=1 
        i+=1

    # for line in temp:
    #     print(line)

    cnt=0 
    for j in range(0, 200):
        if temp[0][j] == temp[1][j] == temp[2][j]==1 : #셋다 겹칠때
            cnt+=1
        else: 
            #둘만 겹칠때 
            if temp[0][j] == temp[1][j] ==1:
                cnt+=1
            if temp[0][j] == temp[2][j] ==1:
                cnt+=1
            if temp[1][j] == temp[2][j] ==1:
                cnt+=1


    #print(cnt)           
    return cnt

print(solution([[0, 1], [2, 5], [3, 9]]))
print(solution([[-1, 1], [1, 3], [3, 9]]))
print(solution([[0, 5], [3, 9], [1, 10]]))


겹치는선분 