# 토큰화: 숫자, 연산자, 괄호로 분해
# 왼쪽부터 차례로 토큰 처리
# 숫자 → 출력 큐에 바로 추가
# 왼쪽 괄호 ( → 연산자 스택에 push
# 오른쪽 괄호 ) → (를 만날 때까지 연산자 pop하여 출력 큐에 추가
# 연산자 → 스택의 top 연산자와 우선순위 비교:
# 낮거나 같으면 pop하여 출력 큐에 추가
# 이후 현재 연산자 push
# 입력이 끝난 후, 스택에 남은 연산자를 출력 큐에 순서대로 pop
from stack import MyStack 

#infix 수식을 받아서 postfix수식으로 전환하여 반환하는 함수 
def isOperator(c):
    return c in "+-/*"

def isSpace(c):
    return c == " "

#우선순위 가져오는 함수 
def getPriority(c):
    if c in "+-":
        return 1 
    elif c in "*/":
        return 2 
    else:
        return 0 


def postfix(expr):
    stack = MyStack(100)

    result=""
    for s in expr:
        #1.피연사자면 출력  456
        if s.isdigit():
            result += s
            
        elif isOperator(s): #연산자일 경우에 
            # and 연산은 1 and 1 => 1
            #  0 and ? = 0 
            # 대부분의 언어들이 and 옆의 수식평가가 False 
            # 뒤에 연산 안함 
            #  1 or ?   
            result += " " 
            while not stack.isEmpty() and getPriority(stack.peek()) >= getPriority(s):
                 result = result + stack.pop()
            stack.push(s) 

        else: #공백일때 
            result += s 

    while not stack.isEmpty():
        result += stack.pop()

    return result 

print( postfix("33*5+2"))