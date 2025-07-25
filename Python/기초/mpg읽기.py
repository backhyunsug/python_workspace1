f = open("mpg.csv", "r")
lines = f.readlines()
f.close() 

lines = lines[1:] #1번방부터 끝까지 복사 
#print(lines[:4])
cyliner_count={}
for line in lines:
    line = line[:len(line)-1]#마지막에 있는 \n 지우기 
    values = line.split(",")
    if values[1] in cyliner_count.keys():
        cyliner_count[values[1]] += 1
    else:
        cyliner_count[values[1]] = 1

print( cyliner_count)