/*
function 함수이름(매개변수리스트){

    .....
    return 
}
*/

function add(x, y){
    return x+y;
}

//1~N까지 더해서 출력하는 함수 
function sigma(limit=10){
    s=0;
    for(i=1; i<=limit; i++){
        s+=i;
    }
    return  s;
}

console.log(add(4,5));
console.log(sigma(10));
/*
함수  - 미리 만들고 메모리 계속 차지하고 있음
함수표현식 - 일시적인 쓰고 버리는 함수, 함수이름이 없다. 
let myFunc = function(매개변수){

}

add = function(x, y){ 
    return x+y;
}

이벤트핸들러 , 마우스 왼쪽 눌렀을때 호출되는 함수

화살표함수(람다), this사용불가 (python에서 self와 같은 역할)
add = (x,y)=> x+y; 

*/
add2 = function(x,y){ 
    return x+y; 
}

add3 = (x,y)=> x+y; 

console.log( add2(10,20));
console.log( add3(10,20));

//검색을 해보자 
let arr=[1,2,3,4,5,6,7,8,9,10];
let result = arr.filter( (a) => a%2==0);
console.log(result);

words = ["rain", "umbrellar", "desk", "note", "assembly", 
         "survey", "flower", "cloud", "hospital", "hammer", "murder"
];

//json배열 
let persons = [
    {name:"홍길동", phone:"010-0000-0001"},
    {name:"임꺽정", phone:"010-0000-0002"},
    {name:"장길산", phone:"010-0000-0003"},
    {name:"강감찬", phone:"010-0000-0004"},
    {name:"서희",  phone:"010-0000-0005"}
];

result = words.filter( w=>w.length>=5);
console.log(result);

//문제1. arr배열에서 3의 배수 찾아내기 
result = arr.filter( e=>e%3==0);
console.log(result);
//문제2. arr배열에서 5보다 큰수 찾아내기
result = arr.filter( e=>e>5);
console.log(result);

//문제3. words에서 단어가 a가 들어가는 단어만 
result = words.filter( e=>e.includes("a"));
console.log(result);

//문제4. words에서 단어가 h나 r로 시작하는 단어만 
result = words.filter( e=>e[0]=="h" || e[0]=="r");
console.log(result);

//문제5. persons 에서 전화번호가 010-0000-0002인 사람 이름 
result = persons.filter( e=> e.phone=="010-0000-0001");
if(result.length>0){
    result.forEach( e=>{
        console.log(e.name, e.phone);
    })
}
//문제6. persons 에서 임꺽정인 사람 전화번호 
persons.filter( e=> e.name=="임꺽정")
       .forEach(e=>console.log(e.name, e.phone));

//문제7. persons 에서 임꺽정, 서희전화번호 
persons.filter( e=> e.name=="임꺽정" || e.name=="서희")
       .forEach(e=>console.log(e.name, e.phone));
