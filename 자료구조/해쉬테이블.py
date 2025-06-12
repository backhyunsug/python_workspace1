"""
해쉬테이블 

값 ====> 컴퓨터 메모리로 맵핑 
  맵핑함수- 해수함수:문자열 -> ascii 코드로 만들어서 다 더한다음에 
                           충돌나면 안돼서 수식을 통해서 특정값이 나오게 한다음에
                           %키의전체개수 나머지 
                   
값 =>  해쉬함수 ==> 해쉬테이블에 저장(특정메모리로 이동한다)
                  bucket 
                  배열만, 배열과 링크드리스트, 링크드리스트로만으로도 구서가능

school   =======>  23432%100  32 -> bucket[32]
rain     =======>  33454%100  54 -> bucket[54]
rainbow  =======>  23412%100  32 -> bucket[12]
bucket의 크기는 전체 key값 개수 1/2 ~1/3
bucket[100]                   

collison - 충돌,  배열   bucket[54] ->    ->     ->   

1.해쉬함수 
school  - 각 단어별로 unicode 만들어서 더해서 총합 구하기 
          파이썬에서 문자의 unicode 는 ord('a')
"""
def getHash(key):
    total=0
    for k in key:
        total += ord(k)
    return total 

print( getHash("a"))
print( getHash("korea"))
print( getHash("school"))

#head = [None]*100 
"""
    해쉬테이블 
    
    head[0] -> ("school"|??)->(|None) 
    head[1] -> ("rain"|??)->(|None)
    head[2] -> ("desk"|??)->("chair"|?)->(|None)

"""

#데이터 타입만 
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None 
    def __str__(self):  #기본 연산자 오버라이딩, 객체를 print할때 이 연산자를 overriding 
                        #해주면 원하는 대로 출력을 해준다.
                        #안하면 객체 주소값 출력 

        return  f"{self.data} {self.next}"
    
class HashTable:
    def __init__(self, cnt=10):
        self.cnt = cnt  #10개보다 작으면 최소한 bucket개수를 10개는 
                        #만들자
        if self.cnt<10:
            self.cnt = 10
        self.bucketList = [None]*cnt
        #앞에 완충장치용으로 노드를 미리 만들어서 붙여놓자
        self.bucketList = [ Node() for i in range(0, self.cnt)]
        for i in range(0, 5):
            print(self.bucketList[i])

    def getHash(self, key): #해쉬함수 붙이기 
        total=0
        for k in key:
            total += ord(k)
        return total%self.cnt     

    #해쉬테이블 구성 함수 => 해쉬테이블을 구축을 해야 검색을 빠르게 
    #처음에 해쉬테이블 구축시간이 많이 걸린다. Mysql join 이 외부에서 inner join, outer join
    #실제로 mysql 안에 옵티마이저 Nested loop join(for문 돌림), Hash join(해쉬테이블)
    #hash join 은 데이터가 대용량리고 배치처리( 일감을 한번에 모아서 처리하는 방식)<-> 온라인처리
    # (실시간처리)  에 유리하다. 보통의 경우는 실시간처리(고객이 통장을 개설해주세요), 이때는
    # nested loop join이 빠르다 
    # trade off : 균형잡기 - 다양한 알고리즘중에 적당한 걸 선택해야 한다.   
    def createTable(self, dataList=None): #생성자에서 해도 되는데 코드가 길어질거 같아서 함수를 별도로 제작
                           #생성자에서 호출하면 된다.
        #외부에서 [("book"), ("school"), ("rain")] 값을 받아온다 
        for item in dataList:
            #1.해쉬값을 가져온다
            key = self.getHash(item)
            #print( item, key)
            #2.Node(key) 를 만들어서 노드의 앞쪽에 붙인다
            bucket = Node(item)
            #3.bucketList의 앞쪽에 붙인다.
            bucket.next = self.bucketList[key].next
            self.bucketList[key].next = bucket 
            
    def printList(self):
        for item in self.bucketList:
            trace = item.next 
            while trace!=None:
                print(trace.data, end="=>")
                trace = trace.next 
            print() 

    def search(self, word):
        #1.키값부터 구한다
        key = self.getHash(word)
        trace = self.bucketList[key].next 
        find =False 
        while trace!=None and not find:
            if trace.data == word: 
                find = True 
            else: 
                trace = trace.next 
        if not find:
            print("Not found")
        else:
            print("found") 

hash = HashTable()
hash.createTable([("school"), ("desk"), ("chair"), ("rain"), ("survey"),
                  ("house"), ("home"), ("doll"), ("python"), 
                  ("java"), ("html"), ("javascript")])
hash.printList()
print()
hash.search("house")


