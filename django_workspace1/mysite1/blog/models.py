from django.db import models

# Create your models here.
#반드시 models.Model울 상속받아야 한다.
#id필드는 자동으로 만든다. 
#ORM - 객체지향식 디비 접근 - 쿼리 만들기 싫음 
class Blog(models.Model):
    title = models.CharField("제목", max_length=200) 
    contents = models.TextField("내용") 
    writer = models.CharField("작성자", max_length=200) 
    wdate = models.DateTimeField("작성일", auto_now=True) 
    hit = models.IntegerField("조회수")

    def __str__(self):
        return f"${self.title} ${self.writer}" 
    