s = "hobby"
print( s.count("b") ) #b가 몇개 있는가 
print( s.count("t") )

#문자의 위치값 
print(s.find("b")) #첫번째 문자 위치값을 반환
print(s.find("k")) #없으면 -1을 반환 인덱싱할때 0번방부터 -1이 유용하지 않다는 의미임 
s = "I like star. red star. blue star. I like star."
print( s.count("star"))
print( s.find("star"))

pos1 = s.find("star") #0번째 방부터 찾기 
print(pos1)

pos2 = s.find("star", pos1+1) #첫번째 star 찾은 그 다음부터 찾아라라
print(pos2)

pos3 = s.find("star", pos2+1) 
print(pos3)

pos4 = s.find("star", pos3+1) 
print(pos4)

pos1 = s.index("like")
print( pos1 )

# pos1  = s.index("love") #단어가 없으면 예외가 발생한다. 오류가 남 
# print( pos1 )

s = ",".join("abcd")
print(s)

#[] 에 전달된 단어들을 처음 "," 이 안에 있는 기호로 묶어서 문장으로 만들어주는 역할을 한다 
# [] - list타입으로 부른다. 여러개를 묶어준다 
s = ",".join(["cherry", "banana", "pear", "grape"])
print(s)

#반대역할 문장을 -> list타입 
words = s.split(",") #하나의 문장을 쪼갠다. => list타입으로 전환한다 
print(words)

print( "hi".upper() )
print( "HI".lower() )

s = "      hi     "
print( "*"+ s+"*" )
print( "*"+ s.lstrip() +"*")
print( "*"+ s.rstrip() +"*")
print( "*"+ s.strip() +"*")

print( "python".isalpha() )
print( "python1".isalpha() )

print( "python".isdigit() )
print( "123".isdigit() )

s = "hello"  # upper 바뀐값을 반환 원래 값은 안바뀜 
print(s.upper(), s )
s = s.upper()
print( s )
