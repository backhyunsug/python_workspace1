
class MyList:
    def  __init__(self, data):
        self.data = list(data)
    
    def __str__(self):
        return f"MyList({self.data})"
    
    # [] 연산자를 재지정하기 
    def __getitem__(self, index):
        if index>=0 and index <len(self.data):
            return self.data[index]
        return None
     
    def __setitem__(self, index, value):
        if index>=0 and index <len(self.data):
            self.data[index]=value

m1 = MyList( (1,2,3,4,5,6) )
print(m1)
print(m1.data[0])
print( m1[0] ) #객체안에 존재하는 배열을 외부에서 인덱스를 통해 접근해보자
m1[0]=10
print(m1)
