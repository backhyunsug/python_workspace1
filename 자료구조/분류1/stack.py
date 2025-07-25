class MyStack:
    def __init__(self, size=10):
        if size<10: 
            self.size=10 #최소 크기를 10으로 하자 
        else:
            self.size = size 
        self.stack=[]
        for i in range(0, self.size):
            self.stack.append(0)
        self.top = -1 
    
    # push 함수 
    # isFull상태가 아니면 top증가시키고 그안에 값넣기 
    def isFull(self):
        if self.size-1 == self.top:
            return True 
        return False  
            
    def push(self, data):
        if self.isFull():
            return 
        self.top += 1 
        self.stack[self.top]= data 

    def print(self):
        i=0 
        while i <=self.top:
            print( self.stack[i], end=" ")
            i+=1 
        print()    

    def isEmpty(self):
        if self.top == -1:
            return True 
        return False 
    
    def peek(self):
        if self.isEmpty():
            return None 
        return self.stack[self.top]
    
    def pop(self):
        if self.isEmpty():
            return None 
        item = self.stack[self.top]
        self.top -= 1 
        return item 