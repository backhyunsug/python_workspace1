import re

pattern = r"^abc"
pattern = r"abc"
pattern = r"abc$"


text = ["abc", "abcd", "abc15", "dabc", "", "s", "I love kabcde"]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )

