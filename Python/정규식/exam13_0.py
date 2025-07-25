import re
pattern=r'하늘'
text="하늘에 비가 오고 있습니다.  어제도 비가 왔고 오늘도 비가 오고 있습니다"
regex=re.compile(pattern)#패턴을 컴파일 시킨다 
result=regex.findall(text)#matiching 이 이루어진 모든 문자열의 리스트를 반환합니다 
print(result)
