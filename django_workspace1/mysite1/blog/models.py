from django.db import models

# Create your models here.
#반드시 models.Model울 상속받아야 한다.
#id필드는 자동으로 만든다. 
#ORM - 객체지향식 디비 접근 - 쿼리 만들기 싫음 
#ORM - 테이블이 너무 많고 join 이 많거나 서브쿼리가 많을때는 
#      시간이 많이 걸리고 어려울수도 있다.
#      테이블이 10개미만의 경우의 프로젝트 생성시 좋다 
# Spring Entity 에 대응되는것이 Model클래스다 
# 이 모델기반의 테이블을 만들고 싶으면 setting.py파일에 
# INSTALLED_APPS = [
#   'blog.apps.BoardConfig', 앱등록을 해야 한다. 
# 파일자체는 앱을 구축하면 자동으로 만들어준다. 

# html -> form -> name="userid"  => views.py 
# userid = request.POST.get("userid")
# username = request.POST.get("username")
# 이 노가다 작업대신에 forms.py파일을 만든다. 
#장고가 html =>직렬화 => forms의 변수에 값을 넣어준다 
#write.html forms.py파일 만들어서 디비 등록까지 
class Blog(models.Model):
    title = models.CharField("제목", max_length=200) 
    contents = models.TextField("내용") 
    writer = models.CharField("작성자", max_length=200) 
    wdate = models.DateTimeField("작성일", auto_now=True) 
    hit = models.IntegerField("조회수")

    def __str__(self):
        return f"${self.title} ${self.writer}" 
    