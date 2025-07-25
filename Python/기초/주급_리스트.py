nameList=[]
work_timeList=[]
per_payList=[]
payList=[]

for i in range(0, 5):
    name = input("이름 : ")
    work_time = int(input("일한시간 : "))
    per_pay = int(input("시간당급여액 "))

    nameList.append(name)
    work_timeList.append(work_time)
    per_payList.append(per_pay)

for i in range(0,5):
    total = work_timeList[i] * per_payList[i]
    payList.append( total )

for i in range(0, 5):
    print(f"{nameList[i]} {work_timeList[i]}  {per_payList[i]}  {payList[i]}")
    
