"""
데이터를 저장하는 구조 
1.선형구조 - 배열과 링크드리스트
  1-1. 배열 
       연속된 메모리 공간을 필요로 한다 
       주기억장치(ram)에 연속된 공간이 없으면 할당을 못한다. 
       100m 필요하면 100m를 줄 수 있어야 한다.
       10,20,40,30 조각-단편화 
       압축을 해서 -> 100m를 만들어서 배당(os가)
       데이터를 접근할때 index를 사용한다.
       본래의 배열은 프로그램 시작전에 메모리크기를 정확하게 지정해야 한다 
       그리고 중간에 바꿀 수 없다. 
       a = []          a[0]=10   print( a[1])
       a.append(1) 
       a.append(2) 
       현재는 배열이라고 부르는것들이 인덱스를 사용한다는 공통점 하나만 남았음
       배열의 장점은 첫시작위치를 알면 다음번 요소가 바로 옆에 있다.
       속도가 빠르다 단점은 융통성이 없다. 필요할때 메모리를 늘릴방법도 없고
       안필요하다고 줄일수도 없다. 미리 크게 확보를 해야 한다. 
       데이터 중간에 끼워넣을때 다른데이터들이 이동을 해야 한다. 
       데이터 중간에서 삭제하거나 끼워넣을때 오버헤드가 발생한다. 

  1-2. 링크드리스트
        데이터와 다음번요소에 대한 주소를 저장한다. 
        목걸이공예- 비즈를 가지고 연결시켜나가는 개념 
        데이터+주소  
        (데이터|주소) -> (데이터|주소) -> (데이터|주소)
        필요한 만큼만 만들어서 사용이 가능하다 
        인덱스 사용 못한다. 데이터 접근이 어렵다 
        데이터 중간에 끼워넣거나 중간에서 삭제가 쉽다. 
        링크드리스트의 구조의 단점은 느리다. 

       (파이썬의 LIST -> 배열과 링크드리스트 두개를 합침
        실제로 클래스를 만들면 클래스내부에 배열을 두고 
        이 내부 배열을 접근하게 하는 수단들 연산자 중복이다 ) 

2.비선형구조 - 트리, 그래프 
       부모와 자식으로 나뉜다. 
        a           - 트리구조를 순회하려면 
      B    C        DFS(깊이우선탐색)- STACK
    D  E  F  G      BFS(너비우선탐색) - 큐 
    
    그래프는 전체 망형태를 말한다.

    각각의 데이터 유형에 따라서 순회방법이 다르다. 
    사용자에게 동일한 접근방법을 제시하자 - 반복자(iterator)
    컬렉션류 - list, dict, tuple, set 등 그밖의 라이브러리들 
    내부데이터 접근방법은 통일 iterator를 제공
"""
a = [10,20,30,40]
print(a[0])
it = iter(a)  #반복자 객체를 반환한다. 
print(next(it))
print(next(it))
for i in a:
    print(i)
b = {"red":"빨간색", "green":"초록색", "blue":"파란색"}
it2 = iter(b)
print( next(it2) )
print( next(it2) )
print( next(it2) )
#반복자 목적 : 사용자가 클래스 내부를 몰라도 동일한 방법으로 접근할 수 있게 
#컬렉션 클래스 설계자들이 공통의 인터페이스를 정의해놓고 구현한것

#반복자 가져오는 또다른 방법
#__시작하는 함수들은 내장함수   
it = a.__iter__() 
print( next(it) ) #반복자의 현재 위치값을 반환하고 반복자가 다음번 요소로 이동
#더이상 읽을 데이터가 없으면 파이썬의 경우에는 StopIteration이라는 예외를 
#발생시킨다. 보통은 직접 이렇게 쓰지 말고 , for문 써라 
for i in a:
    print(i)
"""
for(i=0; i<3;l i++){
    printf("%d\n", a[i])
}
"""
it = a.__iter__()
while True:
    try:
        item = next(it) #현재 가리키는 데이터 반환하고 다음 요소로 이동
        print(item )
    except StopIteration:
        print("이터레이터종료")
        break  #while문 종료

#이런거 가능하게 하기 위해서 반복자라는 개념을 사용했다 
for i in a:
    print(i)
#반복자를 구축하는 방법이 인터페이스냐 연산자중복이냐 
#인터페이스 -> 클래스는 클래스인데 구현부분이 없는 클래스

#인터페이는 실제 구현부분이 없는 함수들의 묶음 
#객체를 못만든다. 
class MyInterface:
    def __init__(self, add=None, sub=None):
        self.add = add #함수 
        self.sub = sub     

class MyCalculator(MyInterface):
    def __init__(self):
        self.add=self.add2
        self.sub=self.sub2 
        
    def add2(self, x, y):
        return x+y 

    def sub2(self, x, y):
        return x-y 

m1 = MyCalculator()
print( m1.add(10, 5))
print( m1.sub(10, 5))


"""
MyInterface iis = new MyCalculator() 


"""