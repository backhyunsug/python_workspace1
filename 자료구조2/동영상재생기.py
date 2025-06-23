def getSeconds(strTime): #03:44  -> 받아가서 초로 바꾸어 반환한다 
    m, s = strTime.split(":")
    m = int(m)
    return int(m)*60 + int(s)

def getTime(seconds): #초 => 문자열로 바꾸는 함수 
    m = seconds//60   #{m:02} :자릿수를 2자리로 채우는데    
    s = seconds%60    #2자리가 안되면 0으로 채워라 
    return f"{m:02}:{s:02}"

#print( getSeconds("03:44"))
def solution(video_len, pos, op_start, op_end, commands):
    answer = ''
    video_len = getSeconds(video_len)
    op_start = getSeconds(op_start)
    op_end = getSeconds(op_end)
    pos = getSeconds(pos)

    if pos>=op_start and pos<=op_end: #오프닝 구간인 경우에 
        pos = op_end  
    
    for cmd in commands:
        if cmd=="next":
            pos+=10
            if pos>video_len-10:
                pos = video_len
        elif cmd=="prev":
            pos-=10
            if pos<0:
                pos=0 

        if pos>=op_start and pos<=op_end: #오프닝 구간인 경우에 
            pos = op_end          

    return getTime(pos)

print(solution( "34:33","13:00","00:55",
               "02:55",	["next", "prev"]) )

print(solution( "10:55",	"00:05",	"00:15",	"06:55",	["prev", "next", "next"] ))
print(solution( "07:22",	"04:05",	"00:15",	"04:07",	["next"] ))

#"06:55"

