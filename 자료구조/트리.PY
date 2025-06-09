
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

if __name__ == "__main__":
    root = TreeNode("A")
    root.left = TreeNode("B")
    root.right = TreeNode("C")
    root.left.left=TreeNode("D")
    root.left.right=TreeNode("E")
    root.right.left=TreeNode("F")
    root.right.right=TreeNode("G")
    
    inorder(root)
    
    #스택이나 큐, 시스템이 제공하는 큐를 쓰자 - 재귀호출 

    
