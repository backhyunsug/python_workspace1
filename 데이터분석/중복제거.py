
import pandas as pd 

data = {
    'passenger_code':['A101', 'A102', 'A103', 'A101', 'A104', 'A101', 'A103'],
    'target':['광주', '서울', '부산', '광주', '대구', '광주', '부산'],
    'price':[25000, 27000, 45000, 25000, 35000, 27000, 45000]
}

df = pd.DataFrame(data)
print(df)

print("중복된데이터")
col = df['passenger_code'].duplicated() #True 나 False의 리스트로 보여준다 
print(col)

print("중복 삭제 후")
df2 = df.drop_duplicates() #삭제된 데이터를 반환 , 전체필드가 완전히 일치하는 데이터만 삭제한다 
print(df2)

print("서브셋 사용하기")
df3 = df.drop_duplicates(subset=['passenger_code'])
print(df3)

print("서브셋 사용하기2")
df3 = df.drop_duplicates(subset=['passenger_code', 'target'])
print(df3)
