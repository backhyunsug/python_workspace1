"""
검색 알고리즘 
1.순차검색(선형검색): 데이터를 첫번째부터 순서대로 읽어서 해당데이터를 
찾을때까지 또는 끝까지 끝까지 가도 없으면 존재하지 않음, 자료구조에서 
실행 타임을 결정할때 빅O표기법  O(n)
... 1
... 2
... 3  O(3)
...
for i in range(0, n):    n+100 n이 무한정커지면 100은 의미가 없어 
    ..........           3n + 100    O(n)
                         n**2 + 3n + 30  O(n**2)
2.색인순차:정렬해서 
3.이분검색
 데이터가 반드시 정렬되어 있어야 한다. 
 내부 데이터가 자주 바뀌면 정렬하는 시간이 더 오래걸려서 항상 빠르다고 할 수는
 없다. 
 데이터를 절반을 쪼개서  왼쪽을 선택할지 오른쪽선택할지 
 데이터를 절반을 쪼개서  왼쪽을 선택할지 오른쪽선택할지 
 데이터를 절반을 쪼개서  왼쪽을 선택할지 오른쪽선택할지 
 데이터를 찾을때까지 두개로 나누는 작업을 한다. (이분검색) - O(logN)
3. 해쉬검색 - 이론상 검색 속도가 O(1) 이다. 속도때문에 
메모리를 미친듯이 차지한다. 구현도 어렵다 
trade off -거래할때 서로 합의점  파이썬의 dict - dictionary의 약자 
HashMap, HashTable, Map, Dictionary
"""
a = [1,2,3,4,5,6,7,8,9,10]
key=5 #찾아야할 값 
find=-1  #정수 변수못찾은 상태
for i in range(0, len(a)):
    if key==a[i]: #찾았다
        find=i 
        break
if find==-1:
    print("not found")
else:
    print(f"{find}번째에 있음")

def myfilter( aList, key):
    for i in range(0, len(a)):
        if key==a[i]: #찾았다
            return i 
    return -1  #for문을 다 끝나도 못찾았다 -1을  반환 

pos = myfilter(a, 4)
print(pos)

a=["red", "green", "blue", "cyan", "gray"]
pos = myfilter(a, "cyan")
print(pos)


a = [{"name":"A", "age":12},
     {"name":"B", "age":11},
     {"name":"C", "age":13},
     {"name":"D", "age":14},
     {"name":"E", "age":15} ]
pos = myfilter(a, {"name":"C", "age":34})
print(pos)

def myfilter2( funcKey, a):
    for i in range(0, len(a)):
        if funcKey(a[i]): #찾았다
            return i 
    return -1  #for문을 다 끝나도 못찾았다 -1을  반환 

pos = myfilter2( lambda x  : x["name"]=="C", a)
print(pos)

"""
정렬 => 데이터베이스를 사용하는 순간 데이터베이스 쿼리에서 
       검색과 정렬을 지원한다
       데이터베이스 못쓰는 상황에 파일을 써야 한다면 직접 구현해야 한다 
    순서대로 데이터를 늘어놓는것 
    오름차순 정렬 - 올라가면서 정렬 작은것부터 큰거순으로 
    내림차순 정렬 - 내려가면서 정렬 큰것부터 작은거순으로 
    select, bubble, quick(엄청빠름)

    select 정렬 오름차순의 경우에 
       5  1   2  4  3
      0번방 데이터가 제일 작다고 가정하자 
      0,1 0,2 0,3 0,4  조건에 위배되면 바꿔치기 

      1  5 2 4 3 
      1번방 , 2,3,4
      1 2 5 4 3  2,(3,4)

      1 2 3 5 4   3 4 비교

      1 2 3 4 5  마지막 
"""
def selectSort(dataList, key):
    #0 ...1~n
    #1 ...2~n
    #2 ...1~n 
    #n-1   n
    #aList = dataList #얕은복사- 주소만 복사 aList와 dataList데이터는 
    #같다
    aList = [x for x in dataList] #컴프리헨션을 이용한 깊은 복사 
    for i in range(0, len(aList)-1):
        print(aList)
        for j in range(i+1, len(aList)):
            if key(aList[i])>key(aList[j]): #더 작은것이 앞에 있어야 한다
                temp = aList[i]
                aList[i] = aList[j]
                aList[j]=temp 
    return aList  #반환

a = [5,1,2,4,3]
b = selectSort(a, key= lambda x:x)
print(a)
print(b)

a = [{"name":"A", "age":12},
     {"name":"C", "age":11},
     {"name":"E", "age":13},
     {"name":"D", "age":14},
     {"name":"B", "age":15} ]
b = selectSort(a, key= lambda x:x["name"])
print(a)
print(b)







