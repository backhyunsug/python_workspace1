#파일을 읽기로 열때는 파일이 존재해야 한다 
f = open("데이터파일.txt", "r")
data = f.read() #파일을 통으로 읽는다
print(data)
f.close() 

f = open("데이터파일3.txt", "r")
data = f.read() #파일을 통으로 읽는다, str 타입으로 
print( type(data))
f.close()  #파일을 연다. 파일포인터 - 파일 읽을 위치값이 맨 뒤에 가 있다 

f = open("데이터파일3.txt", "r")
data = f.readlines() #반환값이 list타입이다 
print(type(data) )
print(data)
f.close() 


f = open("데이터파일3.txt", "r")
line = f.readline() #반환값이 list타입이다 
while line !="":
    print(type(line) )
    print(line)
    line = f.readline()
f.close() 



