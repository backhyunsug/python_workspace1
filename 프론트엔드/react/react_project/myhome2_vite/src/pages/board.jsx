import axios from "axios"
import { useEffect, useState } from "react";
import {Link} from 'react-router-dom';

function Board(){
    const [boardList, setBoardList] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    const loadData=  async ()=>{
        setIsLoading(true); 
        let result = await axios.get("http://127.0.0.1:8000/board/list");
        console.log( result);
        setBoardList(result.data.data);
        setIsLoading(false);
    } 

    //만일에 이미 컴포넌트가 떠 있은 상태에서 특정 변수값만 바꾸면 화면에 다시 렌더링
    //..하고 싶으면  [변수명1,변수명2] .......
    useEffect(()=>{
        loadData();
    }, []);
    
    return(
        <div className="container">
            <h1>Board</h1>
            {
                isLoading?<div>데이터로딩중....</div>:
                <table className="table">
                    <colgroup>
                        <col width="8%"/>
                        <col width="*"/>
                        <col width="14%"/>
                        <col width="14%"/>
                        <col width="8%"/>
                    </colgroup>
                    <thead className='table-dark'>
                        <tr>
                            <th>번호</th>
                            <th>제목</th>
                            <th>글쓴이</th>
                            <th>작성일</th>
                            <th>조회수</th>
                        </tr>
                    </thead>
                    <tbody>
                    {
                        boardList.map( (item, i)=>{
                            return(
                                <tr key={item.id}>
                                    <td>{item.id}</td>
                                    <td><Link to={"/board/view/"+item.id}>{item.title}</Link></td>
                                    <td>{item.writer}</td>
                                    <td>{item.wdate}</td>
                                    <td>{item.hit}</td>
                                </tr>
                            )
                        })
                    }
                    </tbody>
                </table>
            }
        </div>
        )
        
}
export default Board; 