#이름, 국어, 영어, 수학성적을 입력받아 총점과 평균을 구하여출력
#입력 :이름(문자열),국어, 영어, 수학 
#출력 :총점, 평균 
#입력
student_name = input("이름?")
student_kor = int(input("국어?"))
student_eng = int(input("영어?"))
student_mat = int(input("수학?"))
#계산 - 수식 : 수학의 경우는 좌변과 우변을 바꿀 수 있다.
#      프로그램의 경우는 좌변은 언제나 변수만 가능하다 
student_total = student_kor + student_eng + student_mat
student_avg = student_total//3  
#출력
print(student_name, student_total, student_avg)