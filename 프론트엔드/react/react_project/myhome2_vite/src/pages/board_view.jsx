
import { useState  } from "react"
import { useNavigate, useParams } from "react-router-dom"
import axios from "axios";

//함수 : props 또는 {변수명}
function BoardView(props){
    let {id} = useParams(); //json형태로 받아오나봄, 해체
     
    return(
        <div className="container">
            <h1>게시판</h1>

            선택된 id : {id}
        </div>
    )
}

export default BoardView; 