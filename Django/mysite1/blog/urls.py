from django.contrib import admin
from django.urls import path
from . import views #자기폴더에 있는 views.py를 찾아와라 

#config/urls.py파일에 모든 요청을 받아서 분배
#config/usrls.py파일에서 guestbook/urls.py를 찾을 수 있게
#config/urls.py   blog/~~~

# html에서  url 'blog:blog_view' 

app_name="blog" #반드시 필요함 

urlpatterns = [
    path("", views.index),
    path("list2", views.getList2, name="blog_list2"), #json형태로 출력
    path("list", views.getList, name="blog_list"),#html 
    path("view/<int:id>", views.view, name="blog_view"),
    path("write", views.write, name="blog_write"), #html페이지로 이동
    path("save", views.save, name="blog_save"),   #데이터를 받아서 디비에 저장  
]
