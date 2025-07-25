"""
컴퓨터활용능력시험 
이름 필기(400)  워드(200) 스프레드쉬트(200) 프리젠테이션(200)

총점을구하고 등급 800 이상이 A, 800미만 600이상이 B, 600미만 400이상이 C,
400 미만은 D, 제시험요망
클린코드(자바)=>  
"""
#입력
name = input("이름 : ")
write = int(input("필기 : "))
word = int(input("필기 : "))
spread = int(input("스프레드쉬트 : "))
ppt = int(input("프리젠테이션 : "))
#계산 
total = write + word + spread + ppt
if total >=800:
    grade="A등급"
elif total >= 600:
    grade="B등급"
elif total >= 400:
    grade="C등급"
else:
    grade="D등급,재시험요망"
    
print(f'{name} {write} {word} {spread} {ppt} {total} {grade}') 



