class MyStack:
    def __init__(self, size=100):
        self.size = max(10, size)
        self.stack = [0] * self.size
        self.top = -1

    def isFull(self):
        return self.top == self.size - 1

    def push(self, data):
        if not self.isFull():
            self.top += 1
            self.stack[self.top] = data

    def isEmpty(self):
        return self.top == -1

    def peek(self):
        if self.isEmpty():
            return None
        return self.stack[self.top]

    def pop(self):
        if self.isEmpty():
            return None
        item = self.stack[self.top]
        self.top -= 1
        return item

    def clear(self):
        self.top = -1

# 전역 스택 인스턴스 생성
op_stack = MyStack()
eval_stack = MyStack()

def is_operator(c):
    return c in '+-*/%'

def get_priority(c):
    if c in '+-':
        return 1
    elif c in '*/%':
        return 2
    return 0

def is_higher(c1, c2):
    return get_priority(c1) <= get_priority(c2)

def is_digit(c):
    return c.isdigit()

def postfix(output, expr):
    op_stack.clear()
    i = 0
    while i < len(expr):
        if is_digit(expr[i]):
            while i < len(expr) and is_digit(expr[i]):
                output.append(expr[i])
                i += 1
            output.append(' ')
        elif expr[i] == '(':
            op_stack.push(expr[i])
            i += 1
        elif expr[i] == ')':
            while not op_stack.isEmpty() and op_stack.peek() != '(':
                output.append(op_stack.pop())
                output.append(' ')
            op_stack.pop()  # '(' 제거
            i += 1
        elif is_operator(expr[i]):
            while (not op_stack.isEmpty() and is_operator(op_stack.peek()) and
                   is_higher(expr[i], op_stack.peek())):
                output.append(op_stack.pop())
                output.append(' ')
            op_stack.push(expr[i])
            i += 1
        elif expr[i] == ' ':
            i += 1
        else:
            i += 1  # 기타 문자는 무시

    while not op_stack.isEmpty():
        output.append(op_stack.pop())
        output.append(' ')

def get_result(post_expr):
    eval_stack.clear()
    i = 0
    while i < len(post_expr):
        if is_digit(post_expr[i]):
            num = 0
            while i < len(post_expr) and is_digit(post_expr[i]):
                num = num * 10 + int(post_expr[i])
                i += 1
            eval_stack.push(num)
        elif post_expr[i] == ' ':
            i += 1
        elif is_operator(post_expr[i]):
            b = eval_stack.pop()
            a = eval_stack.pop()
            op = post_expr[i]
            if op == '+':
                eval_stack.push(a + b)
            elif op == '-':
                eval_stack.push(a - b)
            elif op == '*':
                eval_stack.push(a * b)
            elif op == '/':
                eval_stack.push(a // b)
            elif op == '%':
                eval_stack.push(a % b)
            i += 1
        else:
            i += 1  # 기타 문자는 무시
    return eval_stack.pop()

# 실행 예시
if __name__ == "__main__":
    s = "4+3*(5-2)+90"
    post = []
    postfix(post, s)
    post_str = ''.join(post).strip()
    print("후위표기식:", post_str)
    print("계산결과:", get_result(post_str))
