class Node:
    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = Node()  # Dummy head
        self.tail = Node()  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def insert_head(self, data):
        new_node = Node(data)
        new_node.next = self.head.next
        new_node.prev = self.head
        self.head.next.prev = new_node
        self.head.next = new_node

    def insert_tail(self, data):
        new_node = Node(data)
        new_node.prev = self.tail.prev
        new_node.next = self.tail
        self.tail.prev.next = new_node
        self.tail.prev = new_node

    def delete_head(self):
        if self.head.next != self.tail:
            node_to_delete = self.head.next
            self.head.next = node_to_delete.next
            node_to_delete.next.prev = self.head

    def delete_tail(self):
        if self.head.next != self.tail:
            node_to_delete = self.tail.prev
            self.tail.prev = node_to_delete.prev
            node_to_delete.prev.next = self.tail

    def delete_all(self):
        self.head.next = self.tail
        self.tail.prev = self.head

    def get_head_position(self):
        if self.head.next == self.tail:
            return None
        return self.head.next

    def get_next(self, node):
        if node.next == self.tail:
            return None
        return node.next

class DIC:
    def __init__(self, word, mean):
        self.word = word
        self.mean = mean

# 실행 예시
if __name__ == "__main__":
    dll = DoublyLinkedList()

    dll.insert_tail(DIC("book", "책"))
    dll.insert_head(DIC("desk", "책상"))
    dll.insert_head(DIC("rainbow", "무지개"))

    #dll.delete_all()

    pos = dll.get_head_position()
    while pos is not None:
        dic = pos.data
        print(dic.word, dic.mean)
        pos = dll.get_next(pos)
