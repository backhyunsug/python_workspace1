/*
var -자바스크립트가 인터프리터 언어라서 굳이 변수 선언을 하지 않아도 된다. 
변수선언을 하려면 var을 사용했었음  
*/

a = 10;
var a; //나중에 변수 선언을 한다.
console.log(a);
  
// b=5;
// let b; 
// console.log(b);

//호이스팅 - 블럭안에 새로운 변수가 생성되었어야 하는데 안되고 있음 그래서 let가 나옴
msg = "hello";
if (true){ //무조건 if문 안에 들어가길 원함 
    var msg="안녕하세요";
}
console.log(msg);
