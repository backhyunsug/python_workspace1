class TreeNode:
    def __init__(self, data):
        self.data = data
        self.leftchild = None
        self.rightchild = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert_node(self, data):
        new_node = TreeNode(data)
        if self.root is None:
            self.root = new_node
            return

        parent = None
        current = self.root

        while current is not None:
            if current.data == data:
                return  # 중복은 삽입하지 않음
            parent = current
            if data < current.data:
                current = current.leftchild
            else:
                current = current.rightchild

        if data < parent.data:
            parent.leftchild = new_node
        else:
            parent.rightchild = new_node

    def make_tree(self, s):
        for ch in s:
            if ch != ' ':
                self.insert_node(ch)

    def inorder_traverse(self, node):
        if node:
            self.inorder_traverse(node.leftchild)
            print(f"{node.data}", end=" ")
            self.inorder_traverse(node.rightchild)

    def preorder_traverse(self, node):
        if node:
            print(f"{node.data}", end=" ")
            self.preorder_traverse(node.leftchild)
            self.preorder_traverse(node.rightchild)

    def postorder_traverse(self, node):
        if node:
            self.postorder_traverse(node.leftchild)
            self.postorder_traverse(node.rightchild)
            print(f"{node.data}", end=" ")

    def search(self, key):
        current = self.root
        count = 0
        while current:
            if current.data == key:
                print(f"{count}  found")
                return
            elif key < current.data:
                current = current.leftchild
            else:
                current = current.rightchild
            count += 1
        print("not found")

    def delete(self, key):
        self.root = self._delete_node(self.root, key)

    def _delete_node(self, node, key):
        if node is None:
            return None

        if key < node.data:
            node.leftchild = self._delete_node(node.leftchild, key)
        elif key > node.data:
            node.rightchild = self._delete_node(node.rightchild, key)
        else:
            # 삭제할 노드를 찾음
            # 1. 자식이 없는 경우
            if node.leftchild is None and node.rightchild is None:
                return None
            # 2. 하나의 자식만 있는 경우
            elif node.leftchild is None:
                return node.rightchild
            elif node.rightchild is None:
                return node.leftchild
            # 3. 자식이 둘 다 있는 경우
            min_larger_node = self._find_min(node.rightchild)
            node.data = min_larger_node.data
            node.rightchild = self._delete_node(node.rightchild, min_larger_node.data)
        return node

    def _find_min(self, node):
        current = node
        while current.leftchild is not None:
            current = current.leftchild
        return current


# 실행 예제
if __name__ == "__main__":
    bst = BinarySearchTree()
    bst.make_tree("I AM KOREAN")
    bst.inorder_traverse(bst.root)
    print()

    bst.search('A')
    bst.search('Z')
    bst.delete('C')  # 없지만 테스트
    bst.inorder_traverse(bst.root)
    print()
    bst.preorder_traverse(bst.root)
    print()
    bst.postorder_traverse(bst.root)
    print()
