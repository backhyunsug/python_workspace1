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

# c# 수업할때 자동적으로 디비커넥션풀 만든다고 
# 로그를 붙일 수 있다. 웹서버가 돌아갈때 쿼리로그를 설정해놓으면 고객이 요청이 올때
# 마다 로그가 계속 보여서 로그가 너무 많이 쌓여서 디버깅때만 보는 걸로 해야 한다. 
# ORM이 대세고, 페이징을 직접만들려면 오래걸렸는데 orm으로 오면서 자동으로 만들어준다 
# SPA : Single Page Application -> 웹페이지가 하나인 어플리케이션 
# react, vuejs, angula, polymer 
# 웹클라이언트 ====================> 웹서버 
#                                  기존의 html을 그대로 불러서 클라이언트에게
#                                  장관님인사말같은 경우는 html로 미리만들어놓는다  
#                                  새로만들어서 장관님의 동향 취업박람회가서 사진 
#                                  찍었음 , 이때는 새로운 문서를 만들어서 보낸다.
#                                  문서 전체가 통으로 
# SPA ->  하나의 html을 만들고 상단이나 왼쪽바를 고정시켜놓고 
# 바뀌는 부분을 계속 다시 그린다.(렌더링) 해서 출력하는 구조 
# 화면변화가 스무스하고 이쁘다. SPA프레임워크랑 백앤드로 쪼개진다. 
# templates가 아예 없어지는 상황이다. SPA 는 무한스크롤이 페이징을 대신한다. 

# 지연생성 : lazy 생성 
# 이전에는 생성자만으로 초기화가 가능하다라고 생각했던 시기가 있었음 
# class A                class B  상호참조 
#      lazy b = B()         a = A()   
# 생성자에서 초기화가 불가능하거나 또는 굳이 생성자에서 하고 싶지 않을때 
# lazy라는 데코레이터 또는 키워드일수도 있어 변수 선언만해놓고 나중에 객체가 채워준다  

def list(request): #데이터 여러개 가져오기
    #데이터 전체를 가져와서 잘라서 쓴다는건데 메모리 문제는 어떻게 해결하지? 
    scoreList = Score.objects.all().order_by('-id') 
    # 최신 데이터가 먼저 오도록 정렬 (옵션)
    #데이터베이스에서 데이터를 전부 가져오게 설계되어 있다. 
    #슬라이싱을 주면 limit 명령어로 전환되어서 가져온다 
    #쿼리가 모든 데이터를 가져와서  여기서 잘라낸다. 50000 개쯤 되면 객체로 못 가져올 가능성이 있다. 

    #1. 전체페이지 개수 : 데이터 전체 개수를 알아내고 그리고 한페이지당 몇개씩 보일지를 확인한다 
    #                  데이터 전체 개수 : 545, 한페이지당 10개씩 보겠다 
    #                  전체페이지수 : 545/10  올림수  54.5 ->올림수 54 - 54페이지이지만 
    #                              54.1,54.2 ,,,,,,,,55.9까지  => 55페이지
    #                              math.ceiling 함수를 쓰면 올림을 해준다      
    #                  << < 1 2 3 4 5 6 7 8 9 10 > >> 
    #                  << < 1 2 3 4 5 6 7 8 9 10 > >> 
    #                   < 이전페이지 이 태그는 비활성화한다. 
    #                   > 다음 페이지로로 이동해야 하니까 활성화
    #                       1 2 3 4 5 6 7 8 9 10  현재 10페이지 상태에서 next를 누르면
    #                       11 12 13 14 15 16 17 18 19 20  구간이 바뀌어야 한다 
    #                  미리 만들어서 누군가 제공을 한다. 
    #                  구간정보도 가지고 있어야 하고 현재페이지도 있어야 하고 
    #                                                                  
    # 1. Paginator 객체 생성 - 장고가 제공한다 
    # 첫 번째 인자: 페이징할 쿼리셋 (scoreList)
    # 두 번째 인자: 한 페이지에 보여줄 객체 수 (예: 10개)
    # Score.objects.all() 디비 데이터 다 들고와서 페이지 지정하면 그중에서 잘라서 보여준다 
    print("데이터 개수 : ", len(scoreList)) 
   
    paginator = Paginator(scoreList, 10) # 맨처음에 한번 실행되면 지연 할당을 해서 
    #페이지 전체 개수에 필요한 쿼리만 실행을 한다. 데이터는 냅두고 
    
    # 2. GET 요청에서 'page' 파라미터 값 가져오기
    # 요청에 'page' 파라미터가 없으면 기본값으로 1페이지를 보여줍니다.
    page_number = request.GET.get('page')#파라미터로 
    #http://127.0.0,1:8000/score/list?page=1 urls.py 파일안에서  list?page=1
    #http://127.0.0,1:8000/score/list?page=2
    #  
    # 3. 해당 페이지의 객체들 가져오기
    # page() 메소드는 해당 페이지의 Page 객체를 반환합니다.
    page_obj = paginator.get_page(page_number) 
    #그 페이지에 해당하는 데이터만 불러온다 
    #실제 데이터는 이때 가져온다 원하는 만큼만 

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
    