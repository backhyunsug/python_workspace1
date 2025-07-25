"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

#각각의 url.py파일은 views.py파일과 연동을 해야 한다. 
from django.contrib import admin
from django.urls import path
from . import views 

app_name="score"  #########반드시 해줘야 한다 

urlpatterns = [
    path("", views.index),   # blog/ 호출된다.  
    path("list", views.list, name="list"),
    path("view/<id>", views.view, name="view"),
    path("write", views.write),
    path("save", views.save)
]
