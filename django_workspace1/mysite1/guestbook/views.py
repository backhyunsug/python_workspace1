from django.shortcuts import render
from django.http import HttpResponse, JsonResponse 
#JsonResponse  - dict => json으로 바꿔서 응답하는 클래스 
# Create your views here.

def index(request):
    return HttpResponse("guestbook");

#GET방식이 파라미터 전달방식1 
#x=1&y=1 - 파라미터 
#http://127.0.0.1:8000/guestbook/test1?x=5&y=7
def test1(request):
    x = request.GET.get("x")
    y = request.GET.get("y")
    return HttpResponse(int(x)+int(y))


#GET방식이 파라미터 전달방식2 
# /4/5 - 파라미터 
#http://127.0.0.1:8000/guestbook/test2/5/7
def test2(request, x, y):
    return HttpResponse(int(x)+int(y))

def test3(request):
    if request.method=="POST":
        x = request.POST.get("x")
        y = request.POST.get("y")
        return HttpResponse(int(x)+int(y))
    else:
        return HttpResponse("Error")


"""
문제1.  http://127.0.0.1:8000/guestbook/sigma/10   1~10까지의 합계 반환하기
문제2.  http://127.0.0.1:8000/guestbook/isLeap?year=2025 
윤년이면 윤년 아니면 윤년이아니다. 
문제3.  http://127.0.0.1:8000/guestbook/calc/add/4/5 이면 더하기 연산결과   
        http://127.0.0.1:8000/guestbook/calc/sub/4/5 빼기
"""

def sigma(request, limit):
    limit = int(limit)
    s=0
    for i in range(1, limit+1):
        s+=i
    return  HttpResponse(s) 

def isLeap(request):
    year = request.GET.get("year")
    year = int(year)
    if year%4==0 and year%100!=0 or year%400==0:
        return HttpResponse("윤년이다")
    else:
        return HttpResponse("윤년이 아니다")
    
def calc(request, opcode, x, y):
    if opcode=="add":
        result = int(x)+int(y) 
    elif opcode=="sub":
        result = int(x)-int(y)
    else:
        result="지원안함"
    return HttpResponse(result)


#디비연결안되서 간단하게 list로 데이터 전달하기 
flowers = ["작약", "목단", "이팝나무", "장미", "국화", "진달래", "철쭉"]

def list(request):
    #html 페이지와 결합하고 싶으면
    #/templates  
    return render(request, "guestbook/guestbook_list.html",
                  {"title":"HTML연동하기", 
                   "flowersList":flowers} )

def write(request):
    #html문서로 단순이동만 한다 
    return render(request, "guestbook/guestbook_write.html")

def save(request):
    flower = request.POST.get("flower")
    return HttpResponse(flower)

def add_write(request):
    #html문서로 단순이동만 한다 
    return render(request, "guestbook/add_write.html")

def add_save(request):
    x = request.POST.get("x")
    y = request.POST.get("y")
    opcode = request.POST.get("opcode")
    result=0
    if opcode=="1":
        result = int(x) + int(y)
        opcode="+"
    elif opcode=="2":
        result = int(x) - int(y)
        opcode="-" 
    elif opcode=="3":
        result = int(x) * int(y)
        opcode="*" 
    elif opcode=="4":
        result = int(x) / int(y)
        opcode="/" 

    return HttpResponse(f"{x} {opcode} {y} = {result}")

#데이터를 주는 경우 - 
def getData(request):
    # json_dumps_params={'ensure_ascii': False}, 한글깨짐문제 해결  
    return JsonResponse({"name":"홍길동", "age":23, "phone":"010-0000-0001"},
                         json_dumps_params={'ensure_ascii': False})

def userinfo(request):
    return render(request, "guestbook/userinfo.html")

#HttpResponse - 그냥 텍스트로 응답할때 - 실제개발, 연습용
#render - html 문서와 + python데이터를 하나로 묶어서 던지면 새로운 html 만들어서 클라이언트로 보낸다
#        이러한 동작을 렌더링이라고 표현한다. 
# JsonResponse -> 데이터를 json형태로 반환한다. ui/ux(html 파트가 별도의 프레임워크로 만들어지는 경우가 많다)
#                  angular, react, vuejs , polymer, nextjs도 있고
#   
#장고가 디비를 연동하려면 -> sqlite3 -> 로컬에서만 사용, 단점 :네트워크 지원을 하지 않는다. 
#conda activate mysite 
#mysql 연동을 해보자    pip install mysqlclient 
#Model클래스를 만들면 마이그레이션 테이블을 만드는 코드를 생성한다.  이걸 바탕으로 디비에 테이블을 생성해야 한다. 





