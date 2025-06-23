def solution(cards1, cards2, goal):
    answer = ''
    i=0
    j=0 
    for k in range(0, len(goal)):
        # and 연산은 양쪽 다 True 일때 True 
        # and 연산은 어느 한쪽이 False이면 다른 한쪽은 굳이 연산할 필요가 없다 
        # shortcut 회로  앞의 수식을 먼저 판단하고 이 수식의 결과가 
        # False이면 뒤의 수식을 건너 뛴다. 
        if i<len(cards1) and cards1[i] == goal[k]:
            i+=1 
            k+=1
        elif j<len(cards2) and cards2[j] == goal[k]:
            j+=1
            k+=1  
        else: #둘다 아니면 순서 어긋남 
            return "No"  
    return "Yes"
    
print(solution(["i", "drink", "water"],
               	["want", "to"],	
                ["i", "want", "to", "drink", "water"])) 
print(solution(["i", "water", "drink"],
                ["want", "to"],
                ["i", "want", "to", "drink", "water"])) 	

