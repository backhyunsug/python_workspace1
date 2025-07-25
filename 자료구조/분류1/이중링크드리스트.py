"""
단일링크드리스트 -> 뒤로 추적은 가능 앞으로 추적은 불가능하다 
앞뒤로 추적할 수 있게  prev와 next 두개의 링크를 갖고 있다 
"""
class Node:
    def __init__(self, data=None):
        self.data = data 
        self.prev = None  #앞의 노드 주소를 간직
        self.next = None  #뒤의 노드 주소를 간직 
    
class DoubleLikedList:
    def __init__(self):
        self.head = Node()  #리스트 전체의 시작을 가리킨다 
        self.tail = Node()  #리스트 전체의 끝을 가리킨다
        self.head.next = self.tail 
        self.tail.next = self.tail
        self.head.prev = self.head
        self.tail.prev = self.head
        # head <=>(|) <=> (|) <=> tail 

    def insertHead(self, data):
        temp = Node(data)
        temp.next = self.head.next
        self.head.next = temp 
        temp.prev = temp.next.prev 
        temp.next.prev = temp  

    def printNext(self):
        t = self.head.next
        while t!=self.tail:
            print(f"{t.data} ", end="")
            t = t.next 
        print()

    def printPrev(self):
        t = self.tail.prev
        while t!=self.head:
            print(f"{t.data} ",end="")
            t = t.prev 
        print()
    
    def insertTail(self, data):
        temp = Node(data)
        temp.prev = self.tail.prev
        self.tail.prev = temp
        temp.prev.next = temp 
        temp.next = self.tail   

    #삭제 
    def deleteHead(self):
        if self.head.next == self.tail:
            return 
        
        self.head.next = self.head.next.next
        self.head.next.prev = self.head

    def deleteTail(self):
        if self.tail.prev ==self.head:
            return 
        self.tail.prev = self.tail.prev.prev
        self.tail.prev.next = self.tail   


dlist = DoubleLikedList()
dlist.insertHead("A")
dlist.insertHead("B")
dlist.insertHead("C")
dlist.insertHead("D")
dlist.insertHead("E")
dlist.insertHead("F")
dlist.insertTail("G")
dlist.insertTail("H")
dlist.deleteTail()

dlist.printNext()
dlist.printPrev()



             