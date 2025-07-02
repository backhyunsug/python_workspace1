import pandas as pd
data = {
    'name':['홍길동', '임꺽정', '장길산', '홍경래', '이상민', '김수경'],
    'kor':[90, 80, 70, 70, 60, 70],
    'eng':[99, 98, 97, 46, 77, 56],
    'mat':[90, 70, 70, 60, 88, 99],
}

df = pd.DataFrame(data) 
print(df.head()) #앞의 다섯명의 데이터만 확인 
print(df.head(10)) #10개 까지 확인 가능 

print("지정한 열만 출력")
print(df['name'])
print(df['kor'])
#print(df['name':'eng']) # X

print(df.loc[:, 'name':'eng']) #행전체 name ~ eng까지 

print(df.loc[:, ['name','eng', 'mat']]) #numpy 

print(df.loc[[1,3,5], ['name', 'kor'] ]) #양쪽 다 슬라이싱이 가능하다 

print(df.iloc[[1,3,5], [0, 2]]) #iloc - index 로 

#분석하다 부분집합 가져와서 상세히 봐야할때 
df2 = df.loc[[0,2,4], ['name', 'eng', 'mat']]
print(df2)

#인덱스 다시 부여 
df2 = df2.reset_index()
print(df2)
print(df)


