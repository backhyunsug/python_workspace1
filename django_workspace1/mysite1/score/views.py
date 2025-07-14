from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.http import HttpRequest, HttpResponse
from django.core.paginator import Paginator # Paginator import
from .models import Score # Score 모델이 있다고 가정
#페이징 ????          <<  < 1 2 3 4 5 6 7 8 9 10 > >>
#디비로 데이터를 가져올때 전부 갖고 오느냐? 못 갖고 온다
#페이징쿼리를 써서 원하는 페이지의 데이터 개수만큼 가져오기 
#mysql 의 경우는 limit 0, 10 
#mysql 의 경우는 limit 10, 10
#mysql 의 경우는 limit 20, 10
#mysql 의 경우는 limit 30, 10
#전체페이개수를 구해야 한다. select count(*) from score_score 
#페이지개수 구하기 totalPage = math.ceil(totalCnt / 10)
#orm을 지원하는 프레임워크들은 Paginator 거의 다 지원한다. 
#page에 대한 정보를 저장할 클래스이다

def index(request):
    return redirect("score:score_list")


def list(request): #데이터 여러개 가져오기
    scoreList = Score.objects.all().order_by('-id') # 최신 데이터가 먼저 오도록 정렬 (옵션)
    # 1. Paginator 객체 생성
    # 첫 번째 인자: 페이징할 쿼리셋 (scoreList)
    # 두 번째 인자: 한 페이지에 보여줄 객체 수 (예: 10개)
    paginator = Paginator(scoreList, 10) # 한 페이지에 10개씩 보여줍니다.
    # 2. GET 요청에서 'page' 파라미터 값 가져오기
    # 요청에 'page' 파라미터가 없으면 기본값으로 1페이지를 보여줍니다.
    page_number = request.GET.get('page')
    # 3. 해당 페이지의 객체들 가져오기
    # page() 메소드는 해당 페이지의 Page 객체를 반환합니다.
    page_obj = paginator.get_page(page_number) 
    # 4. 템플릿으로 전달할 컨텍스트
    context = {
        "page_obj": page_obj, # Paginator가 반환한 Page 객체를 전달 (렌더링에 필요)
        "title": "성적처리",
        # 'scoreList': scoreList, # page_obj를 사용
    }
    return render(request, "score/score_list.html", context)

def view(request, id): #데이터 한개 가져오기
    scoreModel = get_object_or_404(Score, pk=id)
    #데이터 가져오기 get_object_or_404
    return render(request, "score/score_view.html", {'item':scoreModel})

def write(request):
    scoreform = ScoreForm() 
    #form객체를 만들어서 키값이 form이어야 한다 
    #modify-> score_write.html 페이지를 등록으로도 쓰고 수정으로도 쓰려고 한다 
    context ={'form':scoreform, 
              'modify':False} #추가하고자 하는 정보가 있으면 계속 추가하면 된다.
    return render(request, "score/score_write.html",context  )

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


def update(request, id): #데이터저장
    print("id값 : ", id)
    scoreModel = get_object_or_404(Score, pk=id)
    print(scoreModel.name)
    if request.method=="POST":
        scoreform = ScoreForm(request.POST)
        scoreModel = scoreform.save(commit=False)
        scoreModel.total = scoreModel.kor +scoreModel.eng + scoreModel.mat
        scoreModel.avg = scoreModel.total/3 
        scoreModel.wdate = timezone.now() 
        scoreModel.save()
        return redirect("score:score_view", pk=scoreModel.id)
    else:
        form = ScoreForm(instance=scoreModel)
        print("name " , form)
    return render(request, 'score/score_write.html', {'form':form, 'modify':True, 'id':id})
    