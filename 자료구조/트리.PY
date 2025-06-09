
class TreeNode:
    def __init__(self, data=None):
        self.data = data 
        #self.edge=[]
        self.left=None 
        self.right=None 

def inorder(node):
    if node: #None이 아니면의 의미 
        inorder(node.left) 
        print(node.data, end="\t")
        inorder(node.right) 

def preorder(node):
    if node: #None이 아니면의 의미 
        print(node.data, end="\t")
        preorder(node.left) 
        preorder(node.right)

def postorder(node):
    if node: #None이 아니면의 의미 
        postorder(node.left) 
        postorder(node.right)   
        print(node.data, end="\t")     
#클래스를 안만들면 
root = None   
def insertNode(data=None):
    #트리에 추가되기 위해서 노드를 하나 만든다. 
    newNode = TreeNode(data)
    if root ==None:
        root = newNode
    # ABCDEFGHIJKLMN  <--------------- 
    # queue에 root 를 (parent입력하고)
    # queue에서 첫번째 root 가 나오면 root.left 가 None이면 추가 아니면 root.left를 큐에추가 
    # root.right 도 확인 None이면 여리가 붙이면 다시 root.right 를 큐에 넣는다 
    #          
    #내가 루트노드
    #내가 어디에 끼어들어갈지 
     
""" 
큐를 이용한 레벨order - 너비우선탐색 
1.큐를 초기화한다.데큐를 써도 된다. 
2.  무조건 큐에 node하 넣고 
3. 큐가 비어 있지 않은 동안 반복한다. 
    3-1 큐로부터 무조건 하나 가져온다 
    3-2 데이터를 출력한다
    3-3 가지고 온 노드의 left가 None이 아니면 큐에 넣는다 
    3-4 가지고 온 노드의 right가 None이 아니면 큐에 넣는다 
"""
from collections import deque

def levelorder(node): #너비 탐색 
    queue = deque() #큐를 하나 만든다. 
    queue.appendleft( node ) #root 
    while len(queue)!=0: #큐가 empty 때까지
        current = queue.pop() 
        print(current.data, end="\t")
        if current.left != None:
            queue.appendleft( current.left )
        if current.right !=None:
            queue.appendleft( current.right )

if __name__ == "__main__":
    root = TreeNode("A")
    root.left = TreeNode("B")
    root.right = TreeNode("C")
    root.left.left=TreeNode("D")
    root.left.right=TreeNode("E")
    root.right.left=TreeNode("F")
    root.right.right=TreeNode("G")
    
    print("inorder : ", end="\t")
    inorder(root)
    print() 

    print("preorder : ", end="\t")
    preorder(root)
    print() 
    
    print("postorder : ", end="\t")
    postorder(root)
    print() 
    
    print("levelorder : ", end="\t")
    levelorder(root)
    print() 
    
    #스택이나 큐, 시스템이 제공하는 큐를 쓰자 - 재귀호출 

# from collections  import deque 
# q = deque()
# q.appendleft('A')
# q.appendleft('B')
# q.appendleft('C')
# q.appendleft('D')

# print(len(q) )
# print(q.pop())
# print(len(q) )
# print(q.pop())
# print(len(q) )
# print(q.pop())
# print(len(q) )

