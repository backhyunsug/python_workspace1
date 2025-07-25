data = """
park 800905-1049118
kim  700905-1059119
"""

result = []
for line in data.split("\n"):  #우선 라인단위로 나눈다
    word_result = []
    for word in line.split(" "):#공백으로 나눈다 
        if len(word) == 14 and word[:6].isdigit() and word[7:].isdigit():
            word = word[:6] + "-" + "*******"
        word_result.append(word)
    result.append(" ".join(word_result))
print("\n".join(result))


import re 

data = """
park 800905-1049118
kim  700905-1059119
"""
#서식이 맞는 것만 찾는다다
pat = re.compile("(\d{6})[-]\d{7}")
print(pat.sub("\g<1>-*******", data)) #그룹1의 내용을 바꾼다 
