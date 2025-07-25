"""
큐 - 선입선출, 먼저들어온게 먼저 나가는 구조 
     메시지큐, 은행에서 대기줄에 해당된다. 
     버퍼-컴퓨터의 입출력장치와 메모리간에 속도차가 너무 커서 
     일부 메모리 공간을 잘라서 우리가 키보드를 누르면 그 값이 
     버퍼라는 공간에 먼저 들어갔다가 엔터키 누르면 그때 메모리로 들어간다 
     데이터를 모아두었다가 한번에 처리하는 방식 
     배열 - 환형큐 동그라미
     링크드리스트 - 더블링크드리스트
     우선순위큐 - 우선순위를 준다. 우선순위를 따라서 데이터 순서가 뒤바뀐다
              - 메시지큐 : 윈도우 os안에 있음. 사람의 동작(이벤트)가 미친듯
              이 발생하면 컴퓨터가 미처 감당이 안되니까 각각의 이벤트에 
              번호를 붙여서 어디서 무슨일이 있었는지 다 기록해서 큐에 넣어놓는다
      
     기본큐 : 한쪽 방향에서 데이터를 넣기만 하고 한쪽 방향에서는 데이터를 
             가져가기만 한다.            
     양방향큐 - 양쪽에서 데이터를 넣고 빼기가 다 가능하다. 데큐             

     front - 이쪽에서 데이터를 가져간다 
     rear - 이쪽에서 데이터를 추가한다 

     put - 큐에 데이터 넣는 연산
     get - 큐에서 데이터 가져오기 연산
     isFull - 큐가 차면 True 아니면 False
     isEmpty - 큐가 비었는지 True 아니면 False
     peek - 큐의 맨처음값 하나 확인하는 용도 

     0 1 2 3 4 
     0 1 2 3 4 
     0 1 2 3 4 
     front =0  rear=0          front == rear empty상황 
     put('A')                  0 'A' 0  0  0
     front =0  rear=1         
     put('B')                  0 'A' 'B'  0  0
     front =0  rear=2         
     put('C')                  0 'A' 'B' 'C'  0
     front =0  rear=3         
     put('D')                  0 'A' 'B' 'C'  'D'
     front =0  rear=4                
     put('E')   (rear+1) %5 ==front   full 
     get()                  
     front = 1  rear=4         0 0 'B' 'C' 'D'     
     get()
     front = 1  rear=4         0 0 'B' 'C' 'D'     
     get()
     front = 2  rear=4         0 0 0 'C' 'D'     
     get()
     front = 3  rear=4         0 0 0 0 'D'     
     get()
     front = 4  rear=4         0 0 0 0 0
     front==rear Empty     

     put - 큐에 데이터 넣는 연산
     get - 큐에서 데이터 가져오기 연산
     isFull - 큐가 차면 True 아니면 False
     isEmpty - 큐가 비었는지 True 아니면 False
     peek - 큐의 맨처음값 하나 확인하는 용도 
 
"""
class MyQueue:
    def __init__(self, size=10):
        if size<10: #최소 10이상
            size=10 
        self.size = size 
        self.queue = [None] * self.size
        #print(self.queue)
        self.front = 0 
        self.rear = 0
    
    def put(self, data):
        """
        rear 하나 증가시킨다음에 % self.size 로 나머지 구하고 
        그 위치에 데이터 넣기 
        """
        if self.isFull():
            print("queue is full")
            return 
        self.rear = (self.rear+1)%self.size
        self.queue[self.rear]=data

    def isEmpty(self):
        if self.rear==self.front:
            return True 
        return False 
    
    def isFull(self):
        if (self.rear+1)%self.size==self.front:
            return True 
        return False 

    def get(self): 
        """
        front 증가시켜서 그 위치값 반화하기     
        """
        if self.isEmpty():
            print("queue is empty")
            return None
        self.front = (self.front+1)%self.size 
        return self.queue[self.front]
    
    def peek(self):
        """
        front 증가시켜서 그 위치값 반화하기
        front자체는 바꾸면 안된다
        """
        if self.isEmpty():
            print("queue is empty")
            return None
        temp = (self.front+1)%self.size 
        return self.queue[temp]
    
    def print(self):
        print("큐상태")
        temp = self.front%self.size 
        while temp != self.rear:
            temp = (temp+1)%self.size 
            print(self.queue[temp], end="\t")
            #print("front", self.front, "rear", self.rear, temp)
        

"""
은행에 가면 순번뽑기 
1.은행원 - 작업완료 -> 몇번 손님 나오세요
2.고객 - 번호뽑기 -> 대기인원 
"""
class BankNumber:
    def __init__(self):
        self.number=0 #번호가 0 ~~~~~  
        self.cnt=0    #현재 대기 인원수 
        self.queue = MyQueue(500) #큐의 크기 500 

    def menuCustomer(self):
        self.number += 1
        self.cnt+=1 
        self.queue.put(self.number)
        print(f"대기인원 {self.cnt}" )
        print(f"고객님 번호는 {self.number} 입니다")
    
    def menuBanker(self):
        #큐에서 번호 하나를 꺼낸다 
        if self.queue.isEmpty():
            return 
        number = self.queue.get() 
        print(f"{number}번 고객님 창구앞으로 와주세요")
        self.cnt -= 1
        print(f"대기인원 {self.cnt}" )

    def main(self):
        while True:
            sel = input("1.고객 2.은행원 3.종료")
            if sel=="1":
                self.menuCustomer()
            elif sel=="2":
                self.menuBanker()
            elif sel=="3":
                return 
            else: print("쫌")

if __name__ == "__main__":
   bm = BankNumber()
   bm.main()
