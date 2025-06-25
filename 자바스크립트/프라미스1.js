let promise = new Promise(function(resolve, reject){
    sum=0;
    for(i=1; i<=10; i++){
        sum+=i;
    }

    resolve(sum); //return 사용불가 
    //reject("fail");
});  //리턴받는건 값 55가 아니고 Promise객체이다. 

//동기식 함수를 => 비동기식으로 바꿔주는 클래스임 
promise
.then( (response)=>{
    console.log(response);
    response = response*100;
    return response; 
})
.then( (response)=>{
    console.log("프라미스체인", response);
})
.catch(e=>{
    console.log(e);
})
.finally(()=>{
    console.log("completed");
})
console.log(promise);
//일반함수앞에 async 를 붙이면 비동기개체로 전환시켜라
async function sigma(limit=10){
    s=0;
    for(i=1; i<=limit; i++){
        s+=i;
    }
    return s;
}
console.log( "***", sigma(100));
sigma(1000)
.then((r)=>{
    console.log("async", r);
});

//어떤 경우에 비동기로 꼭 처리해야 하는 경우가 있는데 
/*
    비동기
           비동기 
                  비동기 
                        비동기  - 멸망의 피라미드 
    await : 프라미스 객체가 일을 마무리하길 기다린다. 

    await 명령어가 async를 기다리는거는 맞는데 이 구문 자체사 async함수에서만 
    사용가능하다. 
*/

async function main(){
    let result = await sigma(100);
    console.log("결과", result);
    console.log("ending ................");
}
     
main();
