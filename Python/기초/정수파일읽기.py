
f = open("정수.txt", "r")
lines = f.readlines() 
sum=0
for line in lines:
    if "\n" in line:
        line = line[:len(line)-1]
        #"홍길동,90,80,70" - 콤마로 자르면 list타입반환
    print(line)    
    sum += int(line)
f.close()

print("평균", sum/len(lines))

