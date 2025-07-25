import re

text1 = "I like star"
text2 = "starship is beautiful"

pattern = "star"  #match함수는 첫번째 단어만 가능 
print (re.match( pattern, text1)) #None, 뒤에 있어서 안된다.
print (re.match( pattern, text2)) #

matchObj = re.match( pattern, text2)
print(matchObj.group() )
print(matchObj.start() )
print(matchObj.end() )
print(matchObj.span() )
print( text2[:4])
