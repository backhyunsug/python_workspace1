import {useState, useEffect} from 'react'
import {Link, useNavigate} from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css'
import axios from 'axios'


function BoardWrite(){
    //변수를 json형태로 
    const [board, setBoard]= useState({});
    let history = useNavigate();

    const onChange= (e)=>{
        const {value, name} = e.target; //value, name 
        setBoard({...board, [name]:value});  
    }

    const onSubmit =(e)=>{
        //submit 버튼 누르면 => submit 함수 호출 => 페이지 전체를 서버로부터 불러와야 한다. 못하게 막자 
        e.preventDefault(); //submit버튼의본래기능을 막는다 
        console.log(board);
        axios.post("http://127.0.0.1:8000/board/insert", board)
        .then( (json)=>{
            console.log(json);
            history("/board/list");
        })
        .catch(e=>{
            console.log("등록실패");
        })
    }

    return(
        <div>
            <form onSubmit={onSubmit}>
                제목 : <input type="text" name="title" onChange={onChange} /> <br/>
                작성자 : <input type="text" name="writer" onChange={onChange} /> <br/>
                내용 : <input type="text" name="contents" onChange={onChange} /> <br/>
                
                <button>등록</button>
            </form>
        </div>
    )
}

export default BoardWrite;