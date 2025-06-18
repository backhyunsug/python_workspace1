#이분검색 - 배열 
#이진탐색 트리 
#데이터를 넣어서 트리를 만들때 
#순서를 지키면서 만든다.
#이진트리 => left, right 두개의 에지를 넣는다 
# 16 8 9 2 4 12,17, 21, 23       나보다 작은값은 왼쪽으로 나보다 큰값은 오른쪽으로 
#           16
#      8        17 
#    2   9         21
#   4      12         23 

#Dict을 써도 되고 특정 데이터 타입만 저장하고 싶다면 
class Data:
    def __init__(self, num):
        self.num = num 

#이진트리 구축하기 
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left=None 
        self.right=None

class BinarySearchTree:
    def __init__(self):
        self.root = None     
    def insert(self, data):
        #1.root 노드가 있는지 확인 root==None 없으면 root노드를 만들자 
        if not self.root:
            self.root = TreeNode(data)
            return  #끝났음, 바로 함수를 종료한다. 

        #1.내 노드가 들어갈 위치를 찾자 
        parent = None 
        current = self.root #추적해서 들어가다 보면 None인곳에 추가 가능하다 

        while current:   #current가 None이 아닌동안 
            if current.data.num == data.num:
                return #중복데이터 배제 
            parent = current #현재 위치값 저장을 해놓는다. 나중에 parent와 연결을 시킨다 
            #나보다 작은값은 왼쪽으로 나보다 큰값은 오른쪽으로  
            if data.num < current.data.num:
                current = current.left  
            else:
                 current = current.right  
        #터미널 노드는 에지가 없다.  current 값이 None일때까지 왼쪽 오른쪽 움직이면서 
        #찾아간다. 그래서 뒤에 parent가 따라가야 한다. 

        #노드만들어서 Parent에 연결하기 
        newNode = TreeNode(data)
        if data.num < parent.data.num:
            parent.left = newNode 
        else: 
            parent.right = newNode




