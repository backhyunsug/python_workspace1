import re
# ? - 없거나 하나만 있거나 
# + - 패턴이 하나이상 반복
# * - 없거나 하나이상 반복 

patterns=[r"\d?", r"\d+", r"\d*"]
text = ["abc", "1abc", "12abc", "123", "aa12ab"]

for pattern in patterns:
    resultList=[]
    for item in text:
        result = re.search(pattern, item)
        if result == None:
            resultList.append(item+"-X")
        else:
            resultList.append(item+"-O")

    print(resultList)