#과제가 10개를 입력받기
#입력 : num 
#출력 : 짝수합계(even_sum), 홀수합계(odd_sum)
#카운트변수 : i (1..10)까지 카운트를 하자 

even_sum = 0 
odd_sum = 0 

for i in range(0, 4):
    num =  int(input("정수 : "))
    if num %2==0:
        even_sum += num 
    else:
        odd_sum += num 

print(f"짝수의 합계 : {even_sum}" )
print(f"홀수의 합계 : {odd_sum}" )

#conda init powershell 

#console - 키보드와 모니터 
#예전에는 미니