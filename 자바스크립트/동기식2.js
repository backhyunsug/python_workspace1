let fs = require('fs'); //외부모듈 끌고 오기 
try
{
    data = fs.readFileSync('./동기식1.js', 'utf-8');
    console.log(data);
}
catch(e)
{
    console.log(e);
}
console.log("completed");