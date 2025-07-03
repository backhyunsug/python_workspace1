import pandas as pd
 
data = {
    'fruits':['망고', '딸기', '수박', '파인애플'],
    'price':[2500, 5000,10000, 7000],
    'count':[5, 2, 2, 4],
}

df = pd.DataFrame(data)
df.loc[len(df)] = {'fruits':'사과', 'price':'3500', 'count':10}
print(df)

df2 = df.drop("price", axis=1) #axis-축   0-행 1-열  
print("원본")
print(df)
print("열삭제")
print(df2)

df2 = df.drop(0, axis=0)
print("원본")
print(df)
print("행삭제")
print(df2)
df2 = df2.reset_index() #인덱스를 다시 부여해라 
print("인덱스부여후")
print(df2)