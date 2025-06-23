workerList = []
for i in range(0, 10):
    gender = input("성별 :(M, 남자 F:여자 ")
    salary = int(input("연봉"))
    
    workerList.append( {"gender":gender, "salary":salary})

male_count=0;
female_count=0
male_sum=0
female_sum=0 

for w in workerList:
    if w['gender']=="M":
        male_count+=1 
        male_sum+=w["salary"]
    else:
        female_count+=1 
        female_sum+=w["salary"]

if male_count>0:
    print(f"남자 평균 : {male_sum/male_count}")

if female_count>0:
    print(f"여자 평균 : {female_sum/female_count}")
