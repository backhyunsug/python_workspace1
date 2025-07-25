import re
#전화번호만 추출하기
text = """
    phone : 010-0000-0000 email:test1@nate.com
    phone : 010-1111-1111 email:test2@naver.com
    phone : 010-2222-2222 email:test3@gmail.com
    phtone: 02-345-9090  email:eedseisk@dse.netwdsweds
    """
print()
print("--- 전화번호 추출하기 ---")
phonepattern = r"\d{2,3}-\d{3,4}-\d{4}"

matchObj = re.findall( phonepattern, text) #string 으로 보낸다. 

for item in matchObj:
    print( item)

# \b - boundry 경계 - 단어이어야 한다. 경계 탭,공백,줄바꿈 기호등으로 
# 경계가 구분되어야 matching여부를 판단한다 

print("--- 이메일 추출하기 ---") 
emailpattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b"

matchObj = re.findall( emailpattern, text)
for item in matchObj:
    print( item)
print()
