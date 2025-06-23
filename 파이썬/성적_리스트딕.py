scoreList = list()    #scoreList = []

#1. 입력 
for i in range(0,4):
    name = input("이름 : ")
    kor = int(input("국어 : "))
    eng = int(input("영어 : "))
    mat = int(input("수학 : "))

    score={} 
    score["name"]= name 
    score["kor"] = kor 
    score["eng"] = eng 
    score["mat"] = mat

    scoreList.append( score ) 
    

for i in range(0, len(scoreList)):
    score = scoreList[i] #저장되었던 dict타입 객체를 가져와서 
    score["total"] = score["kor"] + score["eng"] + score["mat"]
    score["avg"] = score["total"]//3 #몫만 구하기 

for i in range(0, len(scoreList)):
    score = scoreList[i] #저장되었던 dict타입 객체를 가져와서 
    if score["avg"]>=90:
        score["grade"] = "수"
    elif score["avg"]>=80:   
        score["grade"] = "우"     
    elif score["avg"]>=70:   
        score["grade"] = "미"     
    elif score["avg"]>=60:   
        score["grade"] = "양" 
    else:
        score["grade"] = "가"    


for score in scoreList:
    print( f"{score["name"]}", end="\t")
    print( f"{score["kor"]}", end="\t")
    print( f"{score["eng"]}", end="\t")
    print( f"{score["mat"]}", end="\t")
    print( f"{score["total"]}", end="\t")
    print( f"{score["avg"]}", end="\t")
    print( f"{score["grade"]}")
    

#지방 노동청에 신고가 들어옴 회사가 성별간 임금차별  성별과 연봉이 들어온다.
#직원전체 10명이고, 성별하고 연봉입력받아서 남자 평균 , 여자 평균 구하기 
 