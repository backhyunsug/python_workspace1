words = ["rain", "umbrellar", "desk", "note", "assembly", 
         "survey", "flower", "cloud", "hospital", "hammer", "murder"
];

for(i=0; i<words.length; i++){
    console.log(words[i]);
}

for(i in words){
    console.log(words[i]);
}

for(w of words){
    console.log(w);
}

//람다=>화살표함수 
words.forEach(element => {
    console.log(element);
});

//dict 키와 값 쌍으로 저장하는거 자바스크립트에서는 json이라고 한다 
let person={"name":"홍길동", age:23, phone:'010-0000-0001'};
console.log( person.name);
console.log( person.age);
console.log( person.phone);

console.log( person["name"]);
console.log( person["age"]);
console.log( person["phone"]);

//json배열 
let persons = [
    {name:"홍길동", phone:"010-0000-0001"},
    {name:"임꺽정", phone:"010-0000-0002"},
    {name:"장길산", phone:"010-0000-0003"},
    {name:"강감찬", phone:"010-0000-0004"},
    {name:"서희",  phone:"010-0000-0005"}
];

for(i=0; i<persons.length; i++){
    console.log(`${persons[i].name} ${persons[i].phone}`);
}

for(i in persons){
    console.log(`${persons[i].name} ${persons[i].phone}`);
}

for(p of persons){
    console.log(`${p.name} ${p.phone}`);
}

persons.forEach(p=>{
    console.log(`${p.name} ${p.phone}`);
});


