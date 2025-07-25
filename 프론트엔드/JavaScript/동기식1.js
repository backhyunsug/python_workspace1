
function A(){
    s=0;
    for(i=1; i<=10000000000000; i++){
        s+=i;
    }
    console.log("합계 ", s);
}
function B(){
    for(i=1; i<=10; i++){
        console.log(i);
    }
}

A();
B();