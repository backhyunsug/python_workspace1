let arr = [2,3,4,5,6,7,9,11,13]

//map 매개변수로 전달될 함수는 매개변수가 하나여야 하고 반드시 값을 반환하는 함수여야 한다
result = arr.map( x=> x*2 )
console.log(result);

let words = ["cloud", "rainy", "heavy", "desk", "girl", "ace", "main"];
result = words.map(x => x + " " + x.length);
console.log(result);
result = words.map(x => x.toUpperCase());
console.log(result);

//2차원의 경우 
arr = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
];

//map에 전달할 콜백함수의 매개변수는 원래 3개까지 보통은 2개만 사용한다. 
//배열의 요소, 배열의 인덱스
arr.map( (x, i)=>{
    console.log(i, x);
    return x; //반드시 반환값이 있어야 한다. 
});

result = arr.map( (x, i)=>{
    item = x.map((item, j )=>{
        return item*2;
    });
    return item; //반드시 반환값이 있어야 한다. 
});

console.log( result );


//find, filter와 유사 첫번째값 하나만 반환한다. 
arr = [2,3,4,5,6,7,9,11,13]
result = arr.find( x=>x%2==0); //못찾으면 undefined가 반환된다.
console.log(result);

result = words.filter( x => x.indexOf("a")!=-1);
console.log(result);

result = words.find( x => x.indexOf("a")!=-1);
console.log(result);

//reduce 누적시킬때 사용한다
//arr.reduce(콜백함수, 초기값 );
//arr 의 전체 합계 구하기 

s = arr.reduce( (pre, initvalue)=>{
    return pre+initvalue;
});
console.log(s);

//데이터를 문자열로 바꾸어서 정렬한다. 11 13 2 
arr.sort();
console.log(arr); 

//sort에 전달될 compare함수는 매개변수가 2개가 필요하다. arr[i], arr[j]
//a < b 음수를 반환
//a == b 0을 반환 
// a>b  양수를 반환한다 

arr.sort( (a,b) => parseInt(a) - parseInt(b));
console.log(arr); 

arr.sort( (a,b) => parseInt(b) - parseInt(a));
console.log(arr); 

//본래의 순서는 놔두고 정렬한 데이터를 반환한다 
result = arr.toSorted((a,b) => parseInt(a) - parseInt(b));
console.log(arr);
console.log(result);

var items = [
  { name: "Edward", value: 21 },
  { name: "Sharpe", value: 37 },
  { name: "And", value: 45 },
  { name: "The", value: -12 },
  { name: "Magnetic", value: 13 },
  { name: "Zeros", value: 37 },
];

//문제1. value 속성으로 오름차순으로 정렬하기 
items.sort( (x1, x2)=> parseInt(x1.value) - parseInt(x2.value) );
items.forEach( item=>{
    console.log(item.name, item.value );
});

//문제2. name속성을 오름차순으로 자기순위바뀌지 말고 정렬해서 출력하기 
console.log( "abc" > "abc");
console.log( "abc" == "abc");
console.log( "abc" < "abc");

sortedItem = items.toSorted( (x1, x2)=>{
        if (x1.name > x2.name)
            return -1; 
        else if (x1.name < x2.name)
            return 1;
        else
            return 0;
    });

sortedItem.forEach( item=>{
    console.log(item.name, item.value );
});