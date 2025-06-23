# 5-1  변수에 값 "홍길동,임꺽정,장길산,최영,윤관,강감찬,서희,이순신,남이"
# 5-2 =>list타입으로 전환해서 
# 5-3 =>"서희" 가 몇번째에 있는지 
# 5-4 "이순신", "장영실" 존재여부 확인 
# 5-5 추가할사람 : "정도전", "정약용", "최치원" ....
# 5-6  "서희" => "김종서" 로 바꾸기 
# 5-7 장길산 => 김길산 첫글자만 바꾸기 

names = "홍길동,임꺽정,장길산,최영,윤관,강감찬,서희,이순신,남이"
print( names, type(names))

nameList = names.split(",") #전달된 값으로 문자열을 쪼개서 => list타입으로 반환한다 
print(nameList, len(nameList)) #list, 배열의 길이 

#인덱싱   list, string 경우에 각 요소를 숫자를 통해서 접근 가능하다 
# 0 ,1,2,3,4 ..........
print( nameList[0]) 

#슬라이싱  [시작값:종료값:증감치] 각각 생략 가능하다 
print( nameList[3:]) #3번째 이후로 출력 
print( nameList[:3]) #0부터 3번방 직전까지 출력
print( nameList[::-1] ) #역순으로 
print( nameList[2:5] ) #2,3,4 번방만 출력하기 

print("서희의위치", nameList.index("서희") )
#count함수나 in
if nameList.count("이순신") > 0: #if 조건식의 결과가 0이 아닌 모든것 True , 0은 False
    print("이순신이 존재한다")
else:
    print("이순신이 존재하지 않는다")

if "장영실" in nameList:  #nameList안에 "장영실" 이 존재하면 True
    print("\"장영실\"이 존재한다")
else:
    print("\"장영실\"이 존재하지 않는다")

print(nameList.count("이순신"),  "장영실" in nameList)

nameList.append("정도전")
nameList.append("정약용")
nameList.append("최치원")

nameList.extend(["정도전", "정약용", "최치원"])
print(nameList)

pos = nameList.index("서희") #서희 위치값을 찾아서 반환받는다 
nameList[pos] = "김종서" #그 위치값에 다른 값을 넣는다 

print( nameList)

#장길산 => 김길산 첫글자만 바꾸기   문자열의 경우는 index를 통한 수정이 불가능하다 
#list자체는 되니까 

pos = nameList.index("장길산")
nameList[pos] =  nameList[pos].replace("장", "김")
print( nameList )  

