from django import forms 
from blog.models import Blog 

class BlogForms(forms.ModelForm):
    #Meta 클래스 : 클래스 안에 클래를 설계 
    class Meta:
        #디비에 전달해서 저장할 내용만 
        #fileds에 있는 요소는 html의 form태그안에 name속성이 다 있어야 한다 
        model = Blog 
        fields = ['title', 'writer', 'contents']
        labels = {
            'title':'제목',
            'writer':'작성자',
            'contents':'내용'
        }