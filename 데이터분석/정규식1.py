
import re  

pattern = r"비"
regex = re.compile(pattern)   #re.compile을 해서 패턴을 내부 객체에 등록시킨다 

text = "하늘에 비가 오고 있습니다. 어제도 비가 왔고 오늘도 비가 오고 있습니다."

#패턴에 일치하는 단어 찾기 
result = regex.findall(text)   #일치하는 단어들을 모두 찾아서 list형태로 반환한다 

print(result)


