from django import forms 
from blog.models import Score 

#html-> form -> request.POST.get[""].....
#html ->forms -> model과 연동하는 파일을 만든다 
class ScoreForms(forms.ModelForm):
    class Meta:
        model = Score #모델과 form이 연동된다. 
        #insert into blog_blog(title, writer, contents)....
        fields = ['name', 'kor', 'eng', 'mat']
        labels ={
            'name':"이름",
            'kor':"국어",
            'eng':"영어",
            'mat':"수학",
        }

