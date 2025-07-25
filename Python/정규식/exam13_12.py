import re

pattern = r"\b\d{3}-\d{4}\b" #하이픈 앞에 숫자 3개 이상, 하이픈 뒤에 숫자 4개 이상 
text = ["678-0909", "1234-5678", "0123456789-9999", "111-00000", "", "s", "phone number is 123-3333, email  "]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )

