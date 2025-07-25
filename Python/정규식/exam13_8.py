import re

pattern = r"abc$"
text = ["abc", "dabcd", "asdabc", "d12abc", "", "s"]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item) 
    #match함수는 첫 시작만을 보기때문에 적용안됨
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )