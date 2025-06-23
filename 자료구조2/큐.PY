class CircularQueue:
    def __init__(self, size=13):
        self.QMAX = size + 1  # 실제로는 하나 더 크게 만들어야 full/empty 구분 가능
        self.queue = [None] * self.QMAX
        self.front = 0
        self.rear = 0

    def is_full(self):
        return (self.rear + 1) % self.QMAX == self.front

    def is_empty(self):
        return self.front == self.rear

    def put(self, item):
        if self.is_full():
            print("Queue is Full")
            return
        self.rear = (self.rear + 1) % self.QMAX
        self.queue[self.rear] = item

    def get(self):
        if self.is_empty():
            print("Queue is Empty")
            return None
        self.front = (self.front + 1) % self.QMAX
        return self.queue[self.front]

    def peek(self):
        if self.is_empty():
            print("Queue is Empty")
            return None
        return self.queue[(self.front + 1) % self.QMAX]

# 실행 예제
if __name__ == "__main__":
    q = CircularQueue(size=13)

    for i in range(13):
        q.put(chr(ord('A') + i))

    print("Peek:", q.peek())

    for i in range(13):
        print(q.get(), end="\t")

    print()
