from django.contrib import admin
from django.urls import path
from . import views #자기폴더에 있는 views.py를 찾아와라 

#config/urls.py파일에 모든 요청을 받아서 분배
#config/usrls.py파일에서 guestbook/urls.py를 찾을 수 있게
#config/urls.py   blog/~~~
urlpatterns = [
    path("", views.index),
    path("list/", views.getList)
]
