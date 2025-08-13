import {useState, useEffect} from 'react'
import {Link} from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css'
import axios from 'axios'

function BoardList(){
    //1. 변수를 선언한다. -> state공간에 만들어놔야 한다 
    const [boardList, setBoardList] = useState([]);
    //state공간에 boardList와 setBoardList만들고 번지를 준다.
    const [isLoading, setIsLoading] = useState(false); 
    //프론트 =========> 백앤드 데이터가 덜왔음 오는중에 랜더링이 벌이지면 boardList가 null인 경우 
    //에러메시지 boardList의 map함수가 없어요 이 에러를 방지하기 위해서 
    
    async function loadData(){
        let result = await axios.get('http://127.0.0.1:8000/board/list');
        console.log(result);
        setBoardList(result.data.list); 
    }

    useEffect( ()=>{
        setIsLoading(true); //로딩중 
        loadData();
        setIsLoading(false); //로딩완료 
    }, []); //가장 간단하게, 템플릿들하고 코드랑 마운트될때, 언마운트될때, 또 수시로 앤더링될때 호출된다.  
    

    return(
        <div>
            {
                isLoading? 
                    <div>데이터 로딩중입니다.</div> :
                    <table>
                        {
                            boardList.map((item, i)=>{
                                return(
                                    <tr key={item.id}>
                                        <td>{item.id}</td>
                                        <td>{item.title}</td>
                                        <td>{item.writer}</td>
                                        <td>{item.wdate}</td>
                                        <td>{item.hit}</td>
                                    </tr>
                                ) 
                            })
                        }
                    </table>
            }
        </div>
    )
}

export default BoardList;