#과제가 10개를 입력받기
#입력 : num 
#출력 : 짝수합계(even_sum), 홀수합계(odd_sum)
#카운트변수 : i (1..10)까지 카운트를 하자 

even_sum = 0 
odd_sum = 0 

numberList = [] #리스트 객체를 만든다 
#1.입력부터 
for i in range(0, 4):
    num =  int(input("정수 : "))
    numberList.append( num )

#2.계산하기        
for num in numberList:
    if num %2==0:
        even_sum += num 
    else:
        odd_sum += num 

#3.출력하기
print(f"짝수의 합계 : {even_sum}" )
print(f"홀수의 합계 : {odd_sum}" )
