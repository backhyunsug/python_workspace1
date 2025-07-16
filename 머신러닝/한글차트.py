#차트에 한글지원하기 
#폰트를 바꿔야 하는 
#내컴퓨터에 설치된 한글 글꼴 알아보기 

import matplotlib.font_manager as fm
#글꼴도 ttf방식-true type 글꼴, 트루타입글꼴 이전에는 비트맵방식 
#글꼴을 확대하면 계단모양이 나옴 => 벡터방식전환 - true type 글꼴   
fontlist = [font.name for font in fm.fontManager.ttflist]
print(fontlist)  

import matplotlib.pyplot as plt 
import seaborn as sns 
#차트그리기전에 - seaborn차트도 동일하게 적용된다. 
plt.rcParams['font.family'] = 'Malgun Gothic' # 설치된 나눔고딕 폰트 이름으로 변경
# # 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species') 
plt.suptitle('iris 데이터셋 산포도 행렬')
plt.show() 

