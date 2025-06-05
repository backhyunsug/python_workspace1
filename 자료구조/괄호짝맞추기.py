from stack import MyStack

s = "((a*(b+c))-d) / e"
#한글자씩 읽어서 (와 ) 만 필요 
def isMatch(s):
    stack = MyStack(100)
    for c in s:
        if c=="(":
            stack.push(c)
        elif c==")":
            re = stack.pop()
    if stack.isEmpty() and re!=None:
        return True 
    return False 

print( isMatch(s) )

"""
4 + 5 * 2  - 7/3
(1) 5 * 2 
(2) 7 / 3
(3) 4 + (1)
(4) (3) + (2) 

        - 
     +      /  
   4   *  7 3 
      5 2

  inorder : LDR          4 + 5 * 2 - 7 / 3 
  preorder : DLR         - + 4 * 5 2 / 7 3
  postorder : LRD        4 5 2 * + 7 3 / - 

  1. 숫자면 스택에 push    4 5 2 
  2. 연산자면 두개를 pop   left=5  right=2 연산을 수행해서 결과를 스택에
                         push한다
                         4 10 
     + left = 4 right=10   14 
                         14 7 3 
                         7/3  = 2.33334
                         14 2.3333...
                         14 - 2.3333  11.6666
                         11.666666666                             
"""