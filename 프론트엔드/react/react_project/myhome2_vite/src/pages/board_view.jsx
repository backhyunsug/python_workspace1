
import { useEffect, useState  } from "react"
import { useNavigate, useParams } from "react-router-dom"
import axios from "axios";

//함수 : props 또는 {변수명}
function BoardView(props){
    let {id} = useParams(); //json형태로 받아오나봄, 해체
    const [board, setBoard]=useState({});

    const loadData = async ()=>{
        let result = await axios.get("http://127.0.0.1:8000/board/view/"+id);
        console.log(result.data.data[0]);
        setBoard(result.data.data[0]);
        
    }
    
    useEffect( ()=>{ loadData()}, [])
    return(
        <div className="container">
            <h1>게시판</h1>
            {board.id}<br/>
            {board.title}<br/>
            {board.writer}<br/>
            {board.contents}<br/>
            {board.image_url}<br/>
            {board.image_url && (
                <img src={`http://127.0.0.1:8000/${board.image_url}`} />
            )}
            <br/>
        </div>
    )
}

export default BoardView; 