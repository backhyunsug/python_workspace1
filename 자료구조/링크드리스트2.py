class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class MyList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail 
        self.tail.next = self.tail 

    def insertHead(self, data):
        temp = Node(data)
        temp.next = self.head.next 
        self.head.next = temp 

    def print(self):
        trace = self.head.next 
        while (trace != self.tail):
            print(trace.data)
            trace = trace.next

    def insertTail(self, item):
        temp = Node(item)
       
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = temp

    # 특정 위치에 삽입
    def insertItem(self, index, item):
       
        temp = Node(item)
       
        
m1 = MyList()
m1.insertHead('A')
m1.insertHead('B')
m1.insertHead('C')
m1.insertHead('D')
m1.insertHead('E')
m1.insertHead('F')
m1.print()
