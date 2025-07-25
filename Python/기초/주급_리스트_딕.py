 #한사람분 저장하기 
personList=[
    {"name":"홍길동", "work_time":40, "per_pay":10000},
    {"name":"임꺽정", "work_time":30, "per_pay":20000},
    {"name":"장길산", "work_time":20, "per_pay":30000}
]

for i in range(0,2):
    worker={}
    worker["name"] = input("이름 : ")
    worker["work_time"] = int(input("근무시간 : "))
    worker['per_pay'] = int(input("시간당 급여액 : "))
    personList.append(worker)

for worker in personList:
    worker['pay'] = worker["work_time"] * worker["per_pay"]

for worker in personList:
    print(f"{worker['name']}  {worker['work_time']} {worker['per_pay']} {worker['pay']}")

###### 이름 국어 영어 수학 총점 평균  평균에 대해서  수(90) 우(80) 미(70) 양(60) 가(60미만) 
 