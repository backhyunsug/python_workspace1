#한글처리 cp949-윈도우방식, 표준-utf-8,vscode가 utf-8
f = open("score.txt", "r", encoding="utf-8")
lines = f.readlines() 
for line in lines:
    if "\n" in line:
        line = line[:len(line)-1]
    #print(line)
    words = line.split(",")
    #print(words)
    name = words[0]
    kor = int(words[1])
    eng = int(words[2])
    mat = int(words[3])
    total = kor+eng+mat 
    avg = total/3
    print(name, kor, eng, mat, total,avg )    
    ##################
f.close()

