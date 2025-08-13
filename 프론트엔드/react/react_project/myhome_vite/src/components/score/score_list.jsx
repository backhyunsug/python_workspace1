import { useState, useEffect } from "react";
import {Link} from "react-router-dom"
import axios from "axios";

function ScoreList(){
    const [scoreList, setScoreList]=useState([])
    const [isLoading, setIsLoading]=useState(true);
    
    const loadData=()=>{
        setIsLoading(true);
        //setScoreList([...scoreList, {"name":"홍길동", "kor":90, "eng":90, "mat":90}]);
        axios.get("http://127.0.0.1:8000/score/scoreList")
        .then( (res)=>{
            setScoreList( res.data.scoreList);
        })
        .catch( (error)=>{
            console.log(error);
        })
        .finally(()=>{
            setIsLoading(false);
        }) 
    }

    useEffect(()=>{
        loadData();
    }, [])

    return(
        <div>
            {
                isLoading?<div>loading...</div>:
                <table>
                    {
                        scoreList.map( (item, i)=>{
                            return(
                                <tr>
                                    <td>{item.name}</td>
                                    <td>{item.kor}</td>
                                    <td>{item.eng}</td>
                                    <td>{item.mat}</td>
                                </tr>
                            )
                        })
                    }
                </table>
            }
        <Link className="btn btn-danger" to="/score/insert">
          글쓰기
        </Link>
       
      </div>
        
    )
}

export default ScoreList;