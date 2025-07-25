function myfunc(callback, x, y){
    result = callback(x, y);
    console.log(`${x} ${y} = ${result}`);
}

function add(x, y){
    return x+y;
}

myfunc( add, 8, 7);
myfunc( (x, y)=>x-y, 8, 7);
myfunc( (x, y)=>x*y, 8, 7);

