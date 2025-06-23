"""
#문제1. 주급계산 :이름, 근무시간, 시간당급여액, 추가수당 :근무시간이 20시간을 초과하면 
#               시간당급여액에 50%를 가산한다
#   이름 근무시간 시간당급여   기본금액  수당   총액  
#  홍길동  30     10000     300000  50000  350000

입력 :이름(name), 근무시간(work_time), 시간당급여(hour_pay)
출력 : 기본금(base_pay), 수당(over_pay), 총액(total_pay)
1.입력
2.계산 : 기본금(근무시간*시간당급여), 초과수당은 20시간 넘을때만 
3.출력
"""
name = input("이름 : ")
work_time = int(input("근무시간 : "))
hour_pay = int(input("시간당급여액 : "))

base_pay = work_time * hour_pay
over_pay = 0 
if work_time>20:
    over_pay = (work_time-20)*hour_pay/2 
total_pay = base_pay + over_pay 

print(f"{name} {work_time} {hour_pay} {base_pay} {over_pay} {total_pay}")

