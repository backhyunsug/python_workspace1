import re 

text = "I like star. red star yello star"
pattern="star"
result = re.sub(pattern, "moon", text)
print(result)  #문자열을 전체 체인지 

result = re.sub(pattern, "moon", text, count=2)
print(result)  #문자열을 전체 체인지 
