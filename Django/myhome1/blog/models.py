from django.db import models

#디비와 연동 - 디비스키마를 여기 정의하면 
#그대로 디비테이블을 직접 만들어준다 
#반드시 상속을 받아야 한다 
#ORM기법  객체 관계형 모델 
#자동쿼리 생성 
class Blog(models.Model):
    title = models.CharField("제목", max_length=200)
    contents = models.TextField("내용")
    wdate = models.DateTimeField("작성일", 
                    auto_now_add=True) #객체생성시 자동날짜시간 
    writer = models.CharField("작성자", max_length=40)
    hit = models.IntegerField("조회수")

    #함수오버라이딩 
    def __str__(self):
       return  f"{self.title} {self.contents} {self.writer}"
