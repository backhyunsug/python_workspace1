from django.db import models

# Create your models here.
#무조건 models.Model 을 상속받아야 한다 
#id필드는 자동생성
class Score(models.Model): 
    name = models.CharField("이름", max_length=40)
    kor = models.IntegerField("국어")
    eng = models.IntegerField("영어")
    mat = models.IntegerField("수학")
    total = models.IntegerField("총점")
    avg = models.FloatField("평균")
    wdate = models.DateField("작성일", auto_created=True)
    #python manage.py makemigrations 을 수행하면 디비에 가서 테이블을 
    #생성할 파이썬 코드를 자동으로 생성해준다 
    #그냥 migrations폴더 아래 파일을 지우고 python manage.py makemigrations
    #다시하면 만들어준다 
    