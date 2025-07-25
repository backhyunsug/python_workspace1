//let jsondata = "{bad json}";
//객체리터럴은 '' "" 없거나 다 되는데  json은 키값에 반드시 "" 만 된다.
let jsondata = '{"name":"홍길동","age":23}';
try{
    //데이터 송수신할때 실제로는 json객체를 주고 받는게 아니고 json 형태의 
    //문자열을 주고 받는다. 그래서 파싱작업을 해야 한다. 
    //JSON.parse <=>JSON.stringfy 
    let user = JSON.parse(jsondata);
    console.log(user.name);
}
catch(e){    
    console.log("에러");
}