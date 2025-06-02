#스택-LastInFirstOut - LIFO 구조 
#인터럽트 - 가로채기,  컴퓨터는 한번에 하나만 한다. 
# 읽고 계산하고 출력하고 (사이클)
# 컴퓨터는 지 할일하고(cpu) 계속한다. 입력할게 있으면 인터럽트 
# 출력할거 있으면 인터럽트를 건다.    
# 함수 - 스택 
# 수식트리, 트리순회, 0주소   
# 대기줄, 줄서는거 - 큐구조
# push , pop, peek, isFull, isEmpty
# push - 스택의 마지막에 데이터 넣기 
# pop - 스택의 마지막에 하나 꺼내서 반환하기 
# peek - 스택의 마지막 데이터 반환만 한다 
# isFull - 스택이 꽉 차면 True 아니면 False 반환 
# isEmpty - 스택이 비면 True 아니면 False 반환 
# 내부 데이터를 배열로 둔다 
# [0,0,0,0,0,0,0,0,0]     0 1 2 3 4 5 6 7 8 9 
# top 인덱스 - 스택의 마지막 데이터를 가리킨다. 

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
    

s1 = MyStack()
s1.push('A')
s1.push('B')
s1.push('C')
s1.push('D')
s1.push('E')
s1.push('F')
s1.push('G')
s1.push('H')
s1.push('I')
s1.push('J')
s1.push('K')
s1.push('L')
s1.print() 
print("----------------------")
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())
print( s1.pop())

#스택을 사용해서 문자열뒤집기 
#s2 = MyStack(30)

def reverse(arr):
    s = MyStack(len(arr))
    for i in arr:
        s.push(i)
    
    result =""
    while not s.isEmpty():
        result += s.pop()
    return result 

print( reverse("korea") )