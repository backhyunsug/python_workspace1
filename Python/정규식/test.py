import re
 
pattern = r'비'
text = "하늘에 비가 오고 있습니다.  어제도 비가 왔고 오늘도 비가 오고 있습니다"
regex = re.compile(pattern)
result = regex.findall(text)
print( result )
