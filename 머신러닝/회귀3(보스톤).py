
url="http://lib.stat.cmu.edu/datasets/boston"
#다중회귀분석 : 공분산(특성간에 서로 영향을 주고받는것)을 따져보고 필요없는 특성은 제거하는게 맞다 
#R은 기본적으로 제거해준다. 파이썬은 우리가 해야 한다. 
import pandas as pd   #다양한 유형의 데이터가 있을때 처리 방법 
import numpy as np     
#분리문자가 공백이 아니고 \s+, skiprows = 22줄 건너뛰기  제목줄이 없다. 
df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
print(df.head(10))
