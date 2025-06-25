class Person{
    //생성자 
    constructor(name="", age=0){
        this.name = name;
        this.age = age;
    }

    display(){
        console.log(this.name, this.age);
    }
};

let person = new Person("홍길동", 33);
person.display();


