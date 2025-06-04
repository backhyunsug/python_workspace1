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
    def __init__(self, size):
        if size<10: #최소 10이상
            size=10 
        self.size = size 
        self.queue = [None] * self.size
        print(self.queue)
        self.front = 0 
        self.rear = 0

