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

피연산자(숫자)는 스택에 넣는다. 그럼, 스택의 상태는 [3, 5, 2]이다.
연산자를 만나면 스택에서 피연산자 두 개를 꺼내서 계산한다.
연산자 *를 만났으므로 스택에서 2와 5를 꺼내서 곱한다. 5 * 2 = 10
결괏값을 스택에 저장한다. 이제 스택의 상태는 [3, 10]다.
그다음에 연산자 +가 있으므로 스택에서 10과 3을 꺼내서 더한다. 답은 13이다.                         
"""

s = "4 5 2 * + 7 3 / -"
#한글자 받아가서 숫자인지 확인하는 함수 
def isDigit(ch):
    if(ord(ch)>=ord('0') and ord(ch)<=ord('9')):
        return True 
    return False 

def isOperator(ch):
    if ch=="+" or ch=="-" or ch=="*" or ch=="/":
        return True 
    return False 

def operate(c, left, right):
    if c =="+": 
        result = int(left) + int(right)
    elif c=="-":
        result = int(left) - int(right)
    elif c=="*":
        result = int(left) * int(right)
    else:
        result = int(left) // int(right)  
    return result 

def getResult( postfix): #수식을 받아가서 연산 결과를 반환해야 한다 
    stack = MyStack(100)
    for c in postfix: 
        if isDigit(c):
            stack.push(c)
        elif isOperator(c):
            right = stack.pop()
            left = stack.pop() 
            result = operate(c, left, right)
            stack.push(result)
    return stack.pop()

print( getResult(s))
