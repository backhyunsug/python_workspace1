from collections import deque

class Data:
    def __init__(self, eng, kor):
        self.eng = eng
        self.kor = kor

class TreeNode:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if not self.root:
            self.root = TreeNode(data)
            return

        parent = None
        current = self.root

        while current:
            if current.item.eng == data.eng:
                return  # 중복된 데이터는 삽입하지 않음
            parent = current
            if data.eng < current.item.eng:
                current = current.left
            else:
                current = current.right

        new_node = TreeNode(data)
        if data.eng < parent.item.eng:
            parent.left = new_node
        else:
            parent.right = new_node

    def inorder(self, node=None):
        if node is None:
            node = self.root
            return
        if node:
            self.inorder(node.left)
            print(f"{node.item.eng} {node.item.kor}")
            self.inorder(node.right)

    def level_order(self):
        if not self.root:
            return
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            print(f"{node.item.eng} {node.item.kor}")
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def search(self, key):
        current = self.root
        count = 0
        while current:
            if key.eng == current.item.eng:
                print(f"{count} 번만에찾음")
                return
            count += 1
            if key.eng < current.item.eng:
                current = current.left
            else:
                current = current.right
        print("not found")

    def delete(self, key):
        def delete_node(node, key):
            if not node:
                return None
            if key.eng < node.item.eng:
                node.left = delete_node(node.left, key)
            elif key.eng > node.item.eng:
                node.right = delete_node(node.right, key)
            else:
                # 삭제할 노드를 찾음
                if not node.left and not node.right:
                    return None
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                # 자식이 둘인 경우
                successor_parent = node
                successor = node.right
                while successor.left:
                    successor_parent = successor
                    successor = successor.left
                node.item = successor.item
                if successor_parent != node:
                    successor_parent.left = successor.right
                else:
                    successor_parent.right = successor.right
                print(f"{key.eng} is delete")
            return node

        self.root = delete_node(self.root, key)

# 초기화 및 실행
data_list = [
    Data("red", "빨강"),
    Data("green", "초록"),
    Data("book", "책"),
    Data("desk", "책상"),
    Data("moon", "달"),
    Data("sun", "해"),
    Data("flower", "꽃"),
    Data("fish", "물고기"),
    Data("milk", "우유"),
    Data("meat", "고기"),
    Data("pasta", "파스타"),
]

if __name__ == "__main__":
    bst = BST()
    for d in data_list:
        bst.insert(d)

    print("중위 순회:")
    bst.inorder()
    print("\n레벨 순회:")
    bst.level_order()

    eng = input("찾을 단어 : ")
    bst.search(Data(eng, ""))
