let user = {"student-name":"홍길동", kor:90, eng:80, mat:80};
//키값에 "" or ''를 사용하거나 또는 없어도 된다. 
//student-name  키값안에 특수문자 들어가면 반드시 "" 로 감싸줘야 한다 
console.log(user.kor, user["student-name"]); //uiser["kor"], user.kor

//새로온 필드 추가하기 
user["total"] = user.kor+user.eng+user.mat;
user["avg"]=user.total/3; 

console.log( user );

let students = [
    {name:"A", kor:90, eng:80, mat:90},
    {name:"B", kor:70, eng:80, mat:70},
    {name:"C", kor:80, eng:80, mat:60},
    {name:"D", kor:100, eng:100, mat:100},
    {name:"E", kor:90, eng:80, mat:60} 
];

students.forEach( s=>{
    s.total = s.kor+s.eng+s.mat;
    s.avg=s.total/3;
});

key="B"; 
//문제1. B학생 정보를 찾아서 출력하기 
students.filter( item=> item.name == key)
        .forEach( item=> console.log(item));

//문제2. reduce 사용해서 학생들 전체 평균구하기 

// s = arr.reduce( (pre, initvalue)=>{
//     return pre+initvalue;
// });



//문제3. 총점으로 내림차순 해서 출력하기 