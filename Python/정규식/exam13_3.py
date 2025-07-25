import re

text1 = "I like stars, red star, yellow star"
text2 = "star is beautiful"

print()
pattern = "star"
print (re.search( pattern, text1))
print (re.search( pattern, text2))

matchObj = re.search( pattern, text1)
print(matchObj.group() )
print(matchObj.start() )
print(matchObj.end() )
print(matchObj.span() )

matchObj = re.search( pattern, text2)
print(matchObj.group() )
print(matchObj.start() )
print(matchObj.end() )
print(matchObj.span() )
print()






