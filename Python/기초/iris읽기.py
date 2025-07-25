#한글처리 cp949-윈도우방식, 표준-utf-8,vscode가 utf-8
irisList = [] #1차원데이타가 들어갈 list
f = open("iris.csv", "r", encoding="utf-8")
lines = f.readlines() 

for i in range(1, len(lines)):
    line = lines[i]
    line = line[:len(line)-1]
    print(i, line)
    values = line.split(",")
    data =[float(values[0]), float(values[1]), float(values[2]), 
           float(values[3])]
    irisList.append(data)
f.close()

for iris in irisList:
    print(iris)
#print(irisList)

result = [0,0,0,0]
for j in range(0, 4):
    for i in range(0, len(irisList)):
        result[j] = result[j] + irisList[i][j]

print(result[0]/150, result[1]/150, result[2]/150, result[3]/150)

count = len(irisList)
for i in range(0,4):
    print(f"{result[i]/count:.2f}", end="\t"),
print()

## mpg.csv파일 가져와서 실린더개수 8 6 4  종류별 카운트 하기 