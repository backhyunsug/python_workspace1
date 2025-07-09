from django.shortcuts import render, redirect
from django.utils import timezone
from django.http import HttpRequest, HttpResponse
# Create your views here.


from .models import Score 

def index(request):
    return redirect("score:score_list")

def list(request): #데이터 여러개 가져오기 
    scoreList = Score.objects.all()
    return render(request, "score/score_list.html", 
                  {"scoreList":scoreList, "title":"성적처리"})

def view(request, id): #데이터 한개 가져오기
    return render(request, "score/score_view.html")

def write(request):
    return render(request, "score/score_write.html")

from .models import Score 
from .forms import ScoreForm 

def save(request): #데이터저장
    #csrf - 정상적인 로그인을 납치해가서 다른사이트에서 침입을 한다. 
    #html파일을 get방식으로 부를때 csrf_token을 보내고 있다
    #restpul api - > html없이 데이터만 주고 받을 수 있는 서버 
    if request.method =="POST":
        #name = request.POST.get("name")
        scoreform = ScoreForm(request.POST)
        scoreModel = scoreform.save(commit=False)
        #save를 저장하는 시점에서 form -> model 로 전환되서 온다 
        scoreModel.total = scoreModel.kor +scoreModel.eng + scoreModel.mat
        scoreModel.avg = scoreModel.total/3 
        scoreModel.wdate = timezone.now() 
        scoreModel.save() #프레임워크의 단점은 프로그래머 의사를 제한한다.  
    return redirect("score:score_list")
