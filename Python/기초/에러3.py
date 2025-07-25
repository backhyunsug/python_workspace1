try:
    a = [1,2,3,4,5]
    b = a[5] 
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e)
except Exception as e: # 폭포수(casecading)
    print(e)

#raise "예외문구" - 강제예외발생 
#원래 함수 종료 구문은 return 하는일이 많다. 
#return 값도 전송, 함수가 끝날때 마무리작업을 하고 나온다 
#return은 객체지향 이전부터 존재함
#생성자에 오류가 발생했을때 어떻게 할것이냐? return 사용불가 그래서 만든게 
#raise -> 정리작업도 하고 온다 

class Test:
    def __init__(self):
        #return True
        raise Exception("객체 생성오류") 
try: 
    t1 = Test()
except Exception as e:
    print(e)


