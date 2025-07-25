def add(a, b):
    return a + b

def sub(a, b): 
    return a-b


class Test:
    def __init__(self, x, y):
        self.x = x
        self.y = y 
    
    def display(self):
        print(F"X={self.x} Y={self.y}")
