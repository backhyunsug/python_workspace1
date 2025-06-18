#이분검색 - 배열 
#이진탐색 트리 
#데이터를 넣어서 트리를 만들때 
#순서를 지키면서 만든다.
#이진트리 => left, right 두개의 에지를 넣는다 
# 16 8 9 2 4 12,17, 21, 23       나보다 작은값은 왼쪽으로 나보다 큰값은 오른쪽으로 
#           16
#      8        17 
#    2   10         21
#   4   9   12         23 

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

    def inorder(self, node):
        if node==None:
            return 
        self.inorder(node.left)
        print(node.data.num)
        self.inorder(node.right)

    #데이터 검색함수 
    def search(self, key):
        current = self.root #검색시작위치 
        count = 0 #찾은 회수 
        while current:
            if key.num == current.data.num: #찾았으니까 
                return count 
            count+=1 
            if key.num < current.data.num:
                current = current.left 
            else:
                current = current.right 

        return -1


    def find(self, key):
        parent = None       # 삭제될 노드의 부모와 자식을 연결해야 한다 
        current = self.root # 삭제될 노드의 위치를 찾는다 
        find = False #못찾을 
        while current and not find:
            if current.data.num == key.num:
                #print("***", current.data.num)
                find = True 
            else:
                parent = current 
                if key.num < current.data.num: 
                    current = current.left 
                else:
                    current = current.right 

        #print(find, parent, current , key.num)
        return find, parent, current,  
                
    def delete(self, key):
        #삭제하려고 할 경우에 삭제될 노드를 찾아야 한다. 
        if self.root == None:
            return 
        found, parent, current = self.find(key) 
        
        if found == False: #삭제 대상이 없다. 
            return         
        
        #print( "*******", found, parent, current )
        #return 
        #1. 삭제대상이 자식이 없는 경우 - 그냥 나를 삭제하면 된다. 
        if current.left==None and current.right==None:
            if parent.left ==current: #내가 부모 노드의 왼쪽에 있었다면 
                parent.left=None 
            else:                     #내가 부모의 오른쪽에 있었다면 
                parent.right=None 
            return 
        # not(A or B) => not A and not B  
        if current.left == None and current.right!=None  \
            or current.right== None and current.left!= None:
            if current.left:
                parent.left = current.left
            else:
                parent.right = current.right    
            return 
        
        #자식이 둘다 있을때 트리 전체를 재편한다 
        #삭제될 대상의 오른쪽 서브트리에서 가장 작은 대상을 
        #찾아 바꿔치기를 한다 
        #탐색을 다시 해야 한다 
        subParent = current #삭제 대상 
        subCurrent = current.right 
        print(subCurrent)
        #탐색을 서브트리의 왼쪽으로 왜? 작은 거는 왼쪽으로 큰거는 오른쪽으로 
        while subCurrent.left:
            subParent = subCurrent 
            subCurrent = subCurrent.left  

        current.data=subCurrent.data   #아래의 큰 수를 옮긴다. 
        if subParent.left == subCurrent:
            subParent.left = subCurrent.right 
        else:
             subParent.right = subCurrent.right
    
       

if __name__=="__main__":
    bst = BinarySearchTree()
    arr = [16, 8, 10, 9, 2, 4, 12,17, 21, 23]
    for i in arr:
        bst.insert(Data(i))
    bst.inorder(bst.root)

    print(bst.search(Data(16)))
    print(bst.search(Data(2)))
    print(bst.search(Data(26)))
    
    bst.delete(Data(8))
    bst.inorder(bst.root)
    print(bst.search(Data(8)))

    bst.delete(Data(16))
    #print(bst.search(Data(16)))
    bst.inorder(bst.root)
    






