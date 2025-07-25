let person = {
    name:"홍길동",
    age:23,
    //키에 값만 저장이 아니라 함수도 저장할 수 있다 
    display:function(){
        console.log(`${this.name} ${this.age}`);
        //this - 객체 자신 
        //this생략불가 
    },
    setValue:function(name, age){
        this.name = name;
        this.age = age;
    }
};
    //화살표함수는 this를 접근할 수 없다. 그래서 람다는 불가능, 함수표현식만 가능하다

person.setValue("임꺽정", 33);
person.display();

/*
객체리터럴과 json 
개체리터럴은 개체를 만들고 생성자, 지금처럼 함수도 저장가능하다 
json 은 함수나 생성자 없고, 네트워크를 이용해 정보를 주고받을때 사용ㅎ나다 
데이터 전송용, 데이터만 
*/