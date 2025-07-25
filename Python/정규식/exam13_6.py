import re

text1 = "I like stars, red star, yellow star"

print()
pattern = "star"
result = re.sub( pattern, "moon", text1)
print(result)

result2 = re.sub( pattern, "moon", text1, count=2)
print(result2)
print()

