#문법적으로 javascript 에 JSON 데이터 구조하고 동일 
#MySQL 속도가 너무 느려서 빅데이터 몽고디비-JSON형태로 데이터 저장을 한다 
#판다스 -> 데이터프레임 : dict -> dataframe형태로 바꾸기 쉽다 
workerList = [
    {"name":"홍길동", "work_time":30, "per_pay":20000},
    {"name":"김길동", "work_time":20, "per_pay":30000},
    {"name":"고길동", "work_time":50, "per_pay":20000}
] #공용변수로 존재해야 한다 

def process(worker): #dict가져와서 반환하는 방법,매개변수로 값을 받아오면 외부로 전달은 안된다.  
    worker['pay'] = worker['work_time'] * worker["per_pay"]

def process_main():
    for w in workerList:
        process(w)
    
def append(): #데이터 추가 함수 
    worker={} #dict 타입 객체 생성 
    worker["name"] = input("이름 : ")
    worker["work_time"] = int( input("일한 시간 : "))
    worker["per_pay"] = int( input("시간당 급여액 : "))
    worker["pay"]=0
    workerList.append( worker) #목록에 추가하기 

def output():
    for w in workerList:
        print( f"{w["name"]}", end="\t")
        print( f"{w["work_time"]}", end="\t")
        print( f"{w["per_pay"]}", end="\t")
        print( f"{w["pay"]}", end="\t")
        print() #줄바꿈 코드 
        

# append()
# append()
# output() #함수호출 
def main():
    #return 함수를 종료하면서 함수의 작업 내용을 함수 외부로 전달한다 
    #return 값을 안주면 그냥 함수 종료의 의미이다. 
    while(True): #무한루프 - 종료를 하지 않는다. 하나는 break 문을 쓰거나 return 구문 
        print("1.추가")
        print("2.출력")
        print("3.계산")
        print("0.종료")
        sel = input("선택 : ")
        if sel =="1": #사용자가 1을 입력했을때 
            append() #추가함수 호출하기 
        elif sel=="2":
            output() #출력함수 
        elif sel=="3":
            process_main()
        elif sel=="0":
            print("프로그램을 종료합니다.")
            return #함수 자체를 종료시킨다. 
        else:
            print("잘못 선택하셨습니다.")
        

main()
