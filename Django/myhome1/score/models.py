from django.db import models

# Create your models here.
class Score(models.Model):
    name = models.CharField("이름", max_length=40)
    kor = models.IntegerField("국어")
    eng = models.IntegerField("영어")
    mat = models.IntegerField("수학")
    total = models.IntegerField("총점")
    avg=models.IntegerField("평균")
    wdate = models.DateTimeField("등록일", auto_now_add=True)

    #__str__ overriding 
    def __str__(self):
        return f"{self.name} {self.kor} {self.eng} {self.mat}"  