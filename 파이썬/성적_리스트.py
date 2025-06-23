nameList=[]
korList=[]
engList=[]
matList=[]
totalList=[]
avgList=[]
gradeList=[]

#입력부터 
for i in range(0,4):
    name =  input("이름 : ")
    kor = int(input("국어 : "))
    eng = int(input("영어 : "))
    mat = int(input("수학 : "))

    nameList.append(name)
    korList.append(kor)
    engList.append(eng)
    matList.append(mat)
    
for i in range(0, len(nameList)):
    total = korList[i] + engList[i] + matList[i]
    avg = total/3 
    totalList.append(total)
    avgList.append(avg)

for i in range(0, len(nameList)):
    grade =""
    if avgList[i]>=90 :
        grade = "수"
    elif avgList[i]>=80 :
        grade = "우"
    elif avgList[i]>=70 :
        grade = "미"
    elif avgList[i]>=60 :
        grade = "양"
    else:
        grade="가"
    gradeList.append( grade )

        
for i in range(0, len(nameList)):
    print(f"{nameList[i]}",  end="\t")
    print(f"{korList[i]}",   end="\t")
    print(f"{matList[i]}",   end="\t")
    print(f"{engList[i]}",   end="\t")
    print(f"{totalList[i]}", end="\t")
    print(f"{avgList[i]}",   end="\t")
    print(f"{gradeList[i]}")
    
         