from django.shortcuts import render, redirect 
from django.http import HttpRequest, HttpResponse

# Create your views here.

from .models import Blog
def index(request):
    #데이터를 전부 가져와라 
    return HttpResponse("Hello Django")

def list(request):
    blog_list = Blog.objects.all() 
    return render(request, "blog/blog_list.html", {"blogList":blog_list})

from .forms import BlogForms 
def write(request):
    form = BlogForms() #객체 만들고 
    return render(request, 'blog/blog_write.html')

from django.utils import timezone 
def save(request): 
    #title = request.POST.get("title")
    form = BlogForms(request.POST)
    #반환화는객체가 model임 
    blog = form.save(commit=False)#아직멈춤 title, writer,contents 
    blog.wdate = timezone.now() #wdate, hit 
    blog.hit=0 
    blog.save() 

    #글쓰기하고 나면 목록화면으로 이동한다. 
    #함수를 직접 호출하지 않고 클라이언트를 통해서 다시 목록 요청이 되게 해야 
    #리다이렉트 path("list", views.list, name="list")
    return redirect("blog:list")

def view(request, id):
    print("id", id)
    
    blog=Blog.objects.get(id=id)#디비에서 해당 아이디 읽어오기 
    blog.hit = blog.hit+1  #조회수 증가하기  
    blog.save() 
    #print(blog)
    return render(request, "blog/blog_view.html", {"blog":blog}) 