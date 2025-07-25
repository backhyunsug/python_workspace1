#이름하고 주소를 입력받아서 한문장으로 출력하고 싶다 
#각 변수는 특정타입만 저장하지 않는다. 
# a = 4 
# print(a, type(a)) #type명령어는 변수 a의 타입이 아니라 a가 가리키는 값의 타입 

# a = "test"
# print(a, type(a))

# a = 4.5
# print(a, type(a))

name = input("당신의 이름은? ")
address = input("주소는? ")
#python의 경우에 문자열 + 연산은 문자열 끼리만 가능하다 .
print( name + " 님의 주소는 " + address + " 입니다")