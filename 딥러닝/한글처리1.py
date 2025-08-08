#konlpy - 한글 자연어처리를 도와주는 모듈 , java 라이브러리 
#자바를 먼저 설치하고 환경변수도 등록해야 한다 
#cmd - 관리자권한 
#conda activate mytensorflow 
#pip install  konlpy 
#라이브러리를 이용해서 형태소 분석하기 
from konlpy.tag import Okt   #옛날에 twitter => Okt
text = "한글 자연어 처리는 매우 어렵습니다. ㅜㅜ  네이버 영화평론 분석을 해보겠습니다"

okt = Okt() 
print("형태소 분석")
print(okt.morphs(text)) 

#품사태깅 : 한글을 쪼개서 품사를 준다.  자연어:명사 
print("품사태깅")
print(okt.pos(text)) 

print("명사추출")
print(okt.nouns(text)) 

print("정규화") #줄임말이나 인터넷 용어 변환 
print(okt.normalize(text)) 


print("어간추출") #동사 영용사 원형복원 
print(okt.morphs(text, stem=True)) 


