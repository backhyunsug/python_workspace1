import re 
text1 = "I like star, red star, yellow star"
text2 ="starship is beautiful"

pattern = "star"
print(re.search(pattern, text1)) #none출력됨
print(re.search(pattern, text2)) #none출력됨

matchObject = re.match(pattern, text2)
print( matchObject.group())
print( matchObject.start())
print( matchObject.end())
print( matchObject.span())

print(text2[:4])


