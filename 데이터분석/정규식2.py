import re 

pattern = "\d{5}$"   #영문폰트라서 원화표시가 아니라 역슬래쉬로 보임 

zipcode = input("우편번호를 입력하세요") #2132222dasfdsfds
regex = re.compile(pattern)

#match함수가 패턴이  반드시 시작위치에 있어야 한다. a12234 
result = regex.match(zipcode) #형식이 일치하는게 없으면 None 반환
print(result)
if result==None:
    print("매치하지 않습니다.")
else:
    print("매치합니다")

