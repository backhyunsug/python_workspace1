from django.contrib import admin
from django.urls import path
from . import views

#config/urls.py파일에 모든 요청을 받아서 분배
#config/usrls.py파일에서 guestbook/urls.py를 찾을 수 있게
"""
http://127.0.0.1:8000/guestbook/sigma/10   1~10까지의 합계 반환하기
문제2.  http://127.0.0.1:8000/guestbook/isLeap?year=2025 
윤년이면 윤년 아니면 윤년이아니다. 
문제3.  http://127.0.0.1:8000/guestbook/calc/add/4/5 이면 더하기 연산결과   
        http://127.0.0.1:8000/guestbook/calc/sub/4/5 빼
"""
urlpatterns = [
    path("", views.index),
    path("test1", views.test1),
    path("test2/<x>/<y>", views.test2),
    path("test3", views.test3),
    path("sigma/<limit>", views.sigma),
    #def test2(request, x, y) 변수명이 같아야 함 
    path("isLeap", views.isLeap),
    path("calc/<opcode>/<x>/<y>", views.calc),
    path("list", views.list),
    path("write", views.write),
    path("save", views.save),

    path("add_write", views.add_write),
    path("add_save", views.add_save),    

    #json형식으로 응답 
    path("getData", views.getData),
    path("userinfo", views.userinfo)
    
]
