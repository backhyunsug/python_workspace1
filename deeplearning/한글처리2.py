#정규식으로 사용해서 쓸데 없는 문자들 제거하기 => 영어에서 구두점 제거 
from konlpy.tag import Okt 
import re  #정규식을 사용하기 위한 모듈 
"""
re.sub(r"패턴", "대체할문자열", "내용", count=0)
패턴 : 찾을 정규식 패턴 
"""
def clean_text(text):
    text = text.lower() #대문자를 -> 소문자로 바꿔서 처리하자 
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]", "", text)
    #r -> escape문자를 무력화   \n  -> 줄바꿈 기호  패턴에 \ 를 직접사용할 경우에 escape문자로 인식하면안된다
    #[] -> 필요한 문자열들을 묶어준다. or 
    #[^가-힣] : 한글만  ^ - 를 제외하고 , [^] 제외하고의 의미   ^[] 시작하는 의미 
    #[^가-힣]  : 완성형 한글만 인식함 
    return text

text = "ㅋㅋ AI는 인간의 일을 대체하게 될겁니다. 장점도 있고 단점도 있습니다. ! 인간의 반복적이고 힘든일을 대체했으면 ~"
text = clean_text(text)
print(text)
okt = Okt()
print(okt.morphs(text))

    