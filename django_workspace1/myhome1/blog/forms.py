from django import forms 
from blog.models import Blog 

#html-> form -> request.POST.get[""].....
#html ->forms -> model과 연동하는 파일을 만든다 
class BlogForms(forms.ModelForm):
    class Meta:
        model = Blog #모델과 form이 연동된다. 
        #insert into blog_blog(title, writer, contents)....
        fields = ['title', 'writer', 'contents']
        labels ={
            'title':"제목",
            'writer':"작성자",
            "contents":"내용"
        }

