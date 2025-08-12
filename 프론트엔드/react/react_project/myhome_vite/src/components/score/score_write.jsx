import { useState, useEffect } from "react";
import axios from 'axios';
import { Link, useNavigate  } from "react-router-dom";

function ScoreWrite(){
    const [score, setScore] = useState({name:"", kor:0, eng:0, mat:0});
    let history = useNavigate ();    
    
    //해체 - 모던 스크립트 score -> name, kor,eng 
    const {name, kor, eng, mat } = score; 

    //onChange함수 4개 만들기 귀찮음 
    const onChange= (e)=>{
        const {value, name} = e.target; //value, name 
        //value = e.target.value;
        //name = e.target.name;
        setScore({...score, [name]:value});  
        
    }

    const onSubmit=(e)=>{
        e.preventDefault();  //서버로 전송되는 원래 기능을 막아야하 한다. 
 
        axios.post("http://127.0.0.1:8000/score/insert", score)
        .then((res)=>{
            //alert("등록성공");
            history('/score/list');
            console.log("등록성공");

            //navigate를 사용해서 
            //a 태그, location.href=  X(쓰면안됌), SPA위배 - 페이지 자체를 바꾸기때문에 안된다. 
        })
        .catch((error)=>{
            console.log(error);
        })
    }
    return(
        <div>
            <form onSubmit={onSubmit}>
            이름 : <input type="text" val={name} name="name" onChange={onChange}/> <br/>
            국어 : <input type="text" val={kor}  name="kor" onChange={onChange}/> <br/>
            영어 : <input type="text" val={eng}  name="eng" onChange={onChange}/> <br/>
            수학 : <input type="text" val={mat}  name="mat" onChange={onChange}/> <br/>
            <button type="submit">등록</button>
            </form>
            
        </div>    
         
    ) 
    
}

export default ScoreWrite;