#이름, 시간당급여액, 근무시간
employee_name = input("이름 : ")
work_time = int(input("일한시간 : "))
per_pay = int(input("시간당급여액 : "))

pay = work_time * per_pay 

#대부분의 언어는 문자열과 정수를 더하면 정수를 문자열로 바꿔서 
#문자열 합산연산을 수행, 파이썬은 str 를 사용해서 정수를 문자열로 바꿔줘야 한다
print(employee_name + " 님의 급여는 " + str( pay) + "입니다")
