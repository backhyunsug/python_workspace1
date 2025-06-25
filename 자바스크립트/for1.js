//for1.js 

// i=i+1, i+=1, i++, ++i 
// ++i, i++  : 독자적으로 쓰면 차이가 없다. 
// 다른연산자와 함께 쓸 경우에 문제 있음 
a = 5;
b = ++a;   // =, ++ 연산우선순위 - 전치연산자는 무엇보다 연산우선순위가 높다. 
           // ++a;  a=6    b=a    
console.log(`a=${a} b=${b}`); //둘다 6이 나옴 

a = 5;
b = a++;   // b=a  a++      b=5  a=6
console.log(`a=${a} b=${b}`);

//a = ++a + ++a + a++ + a++;  이런거 하는거 아님 

console.log("1~10까지 출력하기");
/*
    for( 1.변수에 초기값 할당; 2. 조건식; 3.증감치)
    {
        4.수행문
    }

    1
    2 true일때만 동작한다 
    4
    3
    2
    4
    3 

*/
for(i=1; i<=10; i++)
{
    console.log(`i=${i}`);
}

console.log("10~1")
for(i=10; i>0; i--)
{
    console.log(`i=${i}`);
}

console.log("1부터 홀수 10개만");
k=1;
for(i=1; i<=10; i++)
{
    console.log(`k=${k}`);
    k+=2;
}

console.log("1부터 짝수 10개만");
k=2;
for(i=1; i<=10; i++)
{
    console.log(`k=${k}`);
    k+=2;
}

//배열 
let arr = [1,2,3,4,5,6,7,8,9,10];
console.log(arr); //인덱싱만 있음, 슬라이싱 없음 
for(i=0; i<arr.length; i++)
{
    console.log(arr[i])
}

arr.push(11); //데이터추가 
console.log(arr);

//in 연산자 : 배열로부터 index를 가져온다 
for(i in arr){
    console.log(`i=${i} arr[i]=${arr[i]}`);
}

//of 연산자 : 배열로부터 항목을 하나씩 가져온다, 배열 요소를 하나씩 가져온다  
for(i of arr){ 
    console.log(`i=${i}`);
}


